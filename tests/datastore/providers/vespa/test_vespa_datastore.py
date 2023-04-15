
import pytest
from typing import List, Dict
from models.models import (
    DocumentChunkMetadata,
    DocumentMetadataFilter,
    DocumentChunk,
    Query,
    QueryWithEmbedding,
    Source,
)
from vespa.application import Vespa
from vespa.deployment import VespaDocker
from datastore.providers.vespa_datastore import (
    VespaDataStore, VespaConfig, VECTOR_SIZE, app_package
)

def create_embedding(non_zero_pos: int, size: int) -> List[float]:
    vector = [0.0] * size
    vector[non_zero_pos % size] = 1.0
    return vector

@pytest.fixture
def vespa_datastore() -> VespaDataStore:
    instance = VespaDocker()
    client = instance.deploy(app_package)
    yield VespaDataStore(client=client)
    instance.container.stop()
    instance.container.remove()


@pytest.fixture
def initial_document_chunks() -> Dict[str, List[DocumentChunk]]:
    first_doc_chunks = [
        DocumentChunk(
            id=f"first-doc-{i}",
            text=f"Lorem ipsum {i}",
            metadata=DocumentChunkMetadata(),
            embedding=create_embedding(i, VECTOR_SIZE),
        )
        for i in range(4, 7)
    ]
    return {
        "first-doc": first_doc_chunks,
    }


@pytest.fixture
def document_chunks() -> Dict[str, List[DocumentChunk]]:
    first_doc_chunks = [
        DocumentChunk(
            id=f"first-doc_{i}",
            text=f"Lorem ipsum {i}",
            metadata=DocumentChunkMetadata(
                source=Source.email, created_at="2023-03-05", document_id="first-doc"
            ),
            embedding=create_embedding(i, VECTOR_SIZE),
        )
        for i in range(3)
    ]
    second_doc_chunks = [
        DocumentChunk(
            id=f"second-doc_{i}",
            text=f"Dolor sit amet {i}",
            metadata=DocumentChunkMetadata(
                created_at="2023-03-04", document_id="second-doc"
            ),
            embedding=create_embedding(i + len(first_doc_chunks), VECTOR_SIZE),
        )
        for i in range(2)
    ]
    return {
        "first-doc": first_doc_chunks,
        "second-doc": second_doc_chunks,
    }


def test_chunk_to_flat_dict(vespa_datastore, initial_document_chunks):
    flat_dict = vespa_datastore._chunk_to_flat_dict(initial_document_chunks["first-doc"][0])
    assert set(flat_dict.keys()) == {"id", "text", "embedding", "author", "url", "document_id", "source", "source_id", "created_at"}


def test_chunk_list_to_vespa_format(vespa_datastore, initial_document_chunks):
    vespa_batch = vespa_datastore._chunk_list_to_vespa_format(initial_document_chunks["first-doc"])
    assert len(vespa_batch) == 3
    assert isinstance(vespa_batch[0], dict)
    assert set(vespa_batch[0].keys()) == {"id", "fields"}


def test_query_filter_date(vespa_datastore):
    query = QueryWithEmbedding(
        query="lorem",
        top_k=1,
        embedding=[0] * VECTOR_SIZE,
        filter=DocumentMetadataFilter(
            start_date="2000-01-03T16:39:57-08:00", end_date="2010-01-03T16:39:57-08:00"
        ),
    )

    vespa_query = vespa_datastore._query_to_vespa(query)
    assert "created_at >= 946946397" in vespa_query["yql"]


@pytest.mark.asyncio
async def test_upsert(vespa_datastore, initial_document_chunks):
    document_ids = await vespa_datastore._upsert(initial_document_chunks)
    chunks_ingested = vespa_datastore.client.query(body={"yql": "select * from documents where true", "type": "all", "hits": 10})
    print(chunks_ingested)
    assert len(document_ids) == 1
    assert chunks_ingested.number_documents_indexed == 3


@pytest.mark.asyncio
async def test_query_returns_all_on_single_query(vespa_datastore, document_chunks):
    # Fill the database with document chunks before running the actual test
    await vespa_datastore._upsert(document_chunks)

    query_embedding = [0.5, 0.5, 0.5, 0.5, 0.5] + [0.0]*(VECTOR_SIZE-5)
    assert len(query_embedding) == VECTOR_SIZE

    query = QueryWithEmbedding(
        query="lorem",
        top_k=5,
        embedding=query_embedding,
    )
    query_results = await vespa_datastore._query(queries=[query])

    assert 1 == len(query_results)
    assert "lorem" == query_results[0].query
    assert 5 == len(query_results[0].results)


@pytest.mark.asyncio
async def test_delete_removes_by_document_id_filter(
    vespa_datastore,
    document_chunks,
):
    # Fill the database with document chunks before running the actual test
    await vespa_datastore._upsert(document_chunks)

    assert vespa_datastore._n_docs_ingested == 5, "There should be 5 chunks in store after upsert; found {vespa_datastore._n_docs_ingested}"

    await vespa_datastore.delete(
        filter=DocumentMetadataFilter(document_id="first-doc")
    )

    assert vespa_datastore._n_docs_ingested == 2, "There should be 2 chunks in store after delete; found {vespa_datastore._n_docs_ingested}"

@pytest.mark.skip()
@pytest.mark.asyncio
async def test_delete_removes_all(
    vespa_datastore,
    document_chunks,
):
    # Fill the database with document chunks before running the actual test
    await vespa_datastore._upsert(document_chunks)

    await vespa_datastore.delete(delete_all=True)

    assert 0 == vespa_datastore._n_docs_ingested
