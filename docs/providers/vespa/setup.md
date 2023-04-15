# Vespa

[Vespa](https://vespa.ai/) is a fully featured search engine and vector database. It can store documents, vector embeddings and metadata. It can run self-hosted or as a managed service at [Vespa Cloud](https://cloud.vespa.ai/). 

## Self-hosted Vespa Instance

For a minimal self-hosted version using Docker container run

```bash
docker run -p 8080:8080 -p 19071:19071 --name vespa vespaengine/vespa
```

For a production ready deployment refer to the [Vespa documentation](https://docs.vespa.ai/) and export the instance URL as `VESPA_URL` environment variable.

**Example:**

```bash
export VESPA_URL="http://YOUR_HOST.example.com"
```

## Vespa Cloud

WIP! Not tested!

## Running Vespa Integration Tests

A suit of tests verifies the Vespa datastore implementation. To run it, make sure Docker daemon is running.

Then, launch the test suite with this command:

```bash
pytest ./tests/datastore/providers/vespa/test_vespa_datastore.py
```
