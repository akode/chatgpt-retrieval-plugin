import os
from typing import Optional, Dict, List
from datastore.datastore import DataStore
from services.date import to_unix_timestamp
from models.models import (
    DocumentChunk,
    QueryResult,
    QueryWithEmbedding,
    DocumentMetadata,
    DocumentMetadataFilter,
    DocumentChunkWithScore,
)
from pydantic import BaseSettings
import vespa
from vespa.application import Vespa
from vespa.deployment import VespaCloud, VespaDocker
from vespa.package import ApplicationPackage, Field, HNSW, RankProfile


VECTOR_SIZE = 1536

# Create the schema
app_package = ApplicationPackage(name="documents")
app_package.schema.add_fields(
    Field(name="id", type="string", indexing=["attribute", "summary"]),
    Field(
        name="embedding",
        type=f"tensor<float>(x[{VECTOR_SIZE}])",
        indexing=["attribute", "summary", "index"],
        ann=HNSW(
            distance_metric="euclidean",
            max_links_per_node=16,
            neighbors_to_explore_at_insert=500,
        ),
    ),
    Field(
        name="text", type="string", indexing=["summary", "index"], index="enable-bm25"
    ),
    Field(name="document_id", type="string", indexing=["summary", "attribute"]),
    Field(name="source_id", type="string", indexing=["summary", "index"]),
    Field(name="source", type="string", indexing=["summary", "index"]),
    Field(name="url", type="string", indexing=["summary", "index"]),
    Field(name="created_at", type="long", indexing=["summary", "attribute"]),
    Field(name="author", type="string", indexing=["summary", "index"]),
)
app_package.schema.add_rank_profile(
    RankProfile(
        name="vector-similarity",
        inherits="default",
        first_phase="closeness(embedding)",
        inputs=[("query(query_embedding)", f"tensor<float>(x[{VECTOR_SIZE}])")],
    )
)


class VespaConfig(BaseSettings):
    VESPA_URL: str = "http://localhost"
    VESPA_PORT: int = 8080
    VESPA_TENANT: Optional[str] = None
    VESPA_KEY: Optional[str] = None

    class Config:
        env_file = ".env"

    def get_deploy_instance(self):
        if self.VESPA_TENANT is None:
            return VespaDocker.from_container_name_or_id("vespa")
        else:
            return VespaCloud(tenant=self.VESPA_TENANT, key_location=self.VESPA_KEY)

    def get_client(self):
        if self.VESPA_TENANT is None:
            return Vespa(url=self.VESPA_URL)
        else:
            return Vespa(url=self.VESPA_URL, cert=self.VESPA_KEY)


def deploy_schema():
    config = VespaConfig()
    instance = config.get_deploy_instance()
    instance.deploy(app_package)


class VespaDataStore(DataStore):
    def __init__(
        self,
        client,
    ):
        self.client = client
        self.schema = client.application_package.schema.name
        self.tensor_fields = [
            field.name
            for field in self.client.application_package.schema.document.fields
            if "tensor" in field.type
        ]

    def _chunk_to_flat_dict(self, chunk: DocumentChunk) -> Dict[str, any]:
        chunk = chunk.dict()
        metadata = chunk.pop("metadata")
        chunk.update(metadata)
        for field in self.tensor_fields:
            chunk[field] = {"values": chunk[field]}

        chunk["created_at"] = (
            to_unix_timestamp(chunk["created_at"]) if chunk["created_at"] else None
        )
        return chunk

    def _chunk_list_to_vespa_format(self, chunks: List[DocumentChunk]):
        assert isinstance(
            chunks[0], DocumentChunk
        ), "List items should be DocumentChunk"
        batch = [
            {"id": chunk.id, "fields": self._chunk_to_flat_dict(chunk)}
            for chunk in chunks
        ]
        return batch

    @staticmethod
    def _convert_filter(filter: DocumentMetadataFilter) -> Optional[str]:
        """Converts DocumentMetadataFilter to Vespa YQL filter"""
        filters = []
        for field, value in filter.dict().items():
            if value is not None:
                if field == "start_date":
                    filters.append(f"created_at >= {str(to_unix_timestamp(value))}")
                elif field == "end_date":
                    filters.append(f"created_at <= {str(to_unix_timestamp(value))}")
                else:
                    filters.append(f'{field} matches "{str(value)}"')
        return " and ".join(filters)

    def _query_to_vespa(self, query: QueryWithEmbedding) -> Dict[str, any]:
        """Converts QueryWithEmbedding to a Vespa query

        Args:
            query: QueryWithEmbedding

        Returns:
            Vespa query (Dict)
        """
        yql = "select * from sources documents where ({targetHits:20}nearestNeighbor(embedding,query_embedding))"
        if query.filter is not None:
            filter_yql = self._convert_filter(query.filter)
            yql = f"{yql} {filter_yql}"
        return {
            "yql": yql,
            "input.query(query_embedding)": query.embedding,
            "ranking.profile": "vector-similarity",
            "hits": query.top_k,
        }

    def _vespa_result_to_document_chunk_with_score(self, hit) -> DocumentChunkWithScore:
        """Converts a hit returned from Vespa to a DocumentChunkWithScore

        Args:
            hit: Hit returned from Vespa query

        Returns:
            DocumentChunkWithScore
        """
        fields = hit["fields"]
        metadata = DocumentMetadata(**fields)
        return DocumentChunkWithScore(
            id=fields["id"],
            text=fields["text"],
            metadata=metadata,
            embedding=fields["embedding"]["values"],
            score=hit["relevance"],
        )

    @property
    def _n_docs_ingested(self):
        res = self.client.query(
            body={
                "yql": "select * from documents where true",
                "type": "all",
                "hits": 10,
            }
        )
        return res.number_documents_indexed

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """Upsert chunks into the datastore.

        Args:
            chunks (Dict[str, List[DocumentChunk]]): A list of DocumentChunks to insert

        Raises:
            e: Error in upserting data.

        Returns:
            List[str]: The document_id's that were inserted.
        """

        feed_batch = [
            item
            for _, doc in chunks.items()
            for item in self._chunk_list_to_vespa_format(doc)
        ]
        self.client.feed_batch(schema=self.schema, batch=feed_batch)
        return list(chunks.keys())

    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """Query the QueryWithEmbedding against the MilvusDocumentSearch

        Search the embedding and its filter in the collection.

        Args:
            queries (List[QueryWithEmbedding]): The list of searches to perform.

        Returns:
            List[QueryResult]: Results for each search.
        """
        requests = (self._query_to_vespa(query) for query in queries)

        results = self.client.query_batch(requests)
        print(results)
        query_results = [
            QueryResult(
                query=query.query,
                results=[
                    self._vespa_result_to_document_chunk_with_score(item)
                    for item in result.hits
                ],
            )
            for query, result in zip(queries, results)
        ]
        print(query_results)

        return query_results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """Delete the entities based either on the chunk_id of the vector,

        Args:
            ids (Optional[List[str]], optional): The document_ids to delete. Defaults to None.
            filter (Optional[DocumentMetadataFilter], optional): The filter to delete by. Defaults to None.
            delete_all (Optional[bool], optional): Whether to drop the collection and recreate it. Defaults to None.
        """
        if ids is None and filter is None and delete_all is None:
            raise ValueError(
                "Please provide one of the parameters: ids, filter or delete_all."
            )

        if delete_all == True:
            raise NotImplementedError("Deleting all documents is not implemented yet.")

        if filter is not None:
            filter_condition = self._convert_filter(filter)
            yql = (
                f"select id, documentid from sources documents where {filter_condition}"
            )
            qbody = {
                "yql": yql,
                "hits": 10,
                "type": "all",
            }
            res = self.client.query(body=qbody)
            while len((res := self.client.query(body=qbody)).hits) > 0:
                delete_resp = self.client.delete_batch(
                    [{"id": hit["fields"]["id"]} for hit in res.hits]
                )
                if not all(resp.status_code == 200 for resp in delete_resp):
                    break
            return True

        if len(ids) > 0:
            resp = self.client.delete_batch(
                batch=[{"id": id} for id in ids], schema=self.schema
            )
            return resp.status_code == 200


if __name__ == "__main__":
    deploy_schema()
