import argparse
import uuid

from data_loader import embed_texts, load_and_chunk_dataset
from vector_db import DEFAULT_STORAGE_PATH, QdrantStorage

COLLECTION_NAME = "aci_bench_graph_ready"
QDRANT_STORAGE_PATH = DEFAULT_STORAGE_PATH
BATCH_SIZE = 100


def ingest_dataset(replace: bool = False):
    store = QdrantStorage(
        collection=COLLECTION_NAME,
        dim=384,
        path=QDRANT_STORAGE_PATH,
    )
    if replace:
        store.reset_collection()

    documents = load_and_chunk_dataset(allowed_splits={"train"})
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        vectors = embed_texts([document["text"] for document in batch])
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "aci-bench:"
                    f"{document['metadata']['split']}:"
                    f"{document['metadata']['doc_id']}:"
                    f"{document['metadata']['chunk_idx']}",
                )
            )
            for document in batch
        ]
        payloads = [
            {"text": document["text"], **document["metadata"]}
            for document in batch
        ]
        store.upsert(ids, vectors, payloads)
        print(f"Indexed {min(start + len(batch), len(documents))}/{len(documents)} chunks.")

    print(
        f"Ingestion complete: {len(documents)} train chunks in "
        f"Qdrant collection '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index ACI-Bench train notes in a graph-ready Qdrant collection."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and recreate only the graph-ready collection before indexing.",
    )
    args = parser.parse_args()
    ingest_dataset(replace=args.replace)