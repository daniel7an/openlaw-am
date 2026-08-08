"""Embed data/chunks.jsonl and load it into Weaviate.

Vectors are computed client-side (D3: BYO vectors keep the model swappable) with
the model in OPENLAW_EMBED_MODEL. If data/embeddings/ holds precomputed vectors —
e.g. produced on a GPU box — they are used instead of encoding locally.

⚠️ This is an e5-family model: the "passage: " / "query: " prefixes are mandatory.
Dropping them costs a lot of retrieval accuracy, silently.

Usage:
    uv run python index.py                 # embed + (re)load the collection
    uv run python index.py --query "..."   # retrieval smoke test (PLAN step 1.4)
"""
import json
import os
import sys
from pathlib import Path

CHUNKS = Path("data/chunks.jsonl")
EMB_DIR = Path("data/embeddings")
COLLECTION = "Article"

MODEL = os.getenv("OPENLAW_EMBED_MODEL", "Metric-AI/armenian-text-embeddings-2-large")
HOST = os.getenv("OPENLAW_WEAVIATE_HOST", "localhost")
PORT = int(os.getenv("OPENLAW_WEAVIATE_PORT", "8080"))
DIM = 1024


def passage(chunk: dict) -> str:
    """Exactly what gets embedded. Must match the GPU-side script byte for byte."""
    return f"passage: {chunk['title']}\n{chunk['text']}"


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line]


def encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL)


def vectors_for(chunks: list[dict]):
    """Precomputed vectors if present and aligned, else encode locally."""
    import numpy as np

    npy, ids_file = EMB_DIR / "embeddings.npy", EMB_DIR / "ids.json"
    if npy.exists() and ids_file.exists():
        vecs = np.load(npy)
        ids = json.loads(ids_file.read_text())
        if ids == [c["id"] for c in chunks] and vecs.shape == (len(chunks), DIM):
            print(f"  using precomputed vectors {vecs.shape} from {npy}")
            return vecs
        print(f"  ⚠️  {npy} does not match chunks.jsonl (order/shape) — re-encoding locally")

    print(f"  encoding {len(chunks)} chunks with {MODEL} (no GPU vectors found)")
    return encoder().encode(
        [passage(c) for c in chunks], normalize_embeddings=True, batch_size=16, show_progress_bar=True
    )


def connect():
    import weaviate

    return weaviate.connect_to_local(host=HOST, port=PORT)


def build() -> None:
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances

    chunks = load_chunks()
    print(f"  {len(chunks)} chunks from {CHUNKS}")
    vecs = vectors_for(chunks)

    client = connect()
    try:
        # Idempotent: the collection is rebuilt from scratch on every run.
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)
        client.collections.create(
            COLLECTION,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="cite_id", data_type=DataType.TEXT),
                Property(name="docid", data_type=DataType.TEXT),
                Property(name="slug", data_type=DataType.TEXT),
                Property(name="article", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="repealed", data_type=DataType.BOOL),
                Property(name="has_repealed_parts", data_type=DataType.BOOL),
            ],
        )

        coll = client.collections.get(COLLECTION)
        with coll.batch.fixed_size(batch_size=100) as batch:
            for chunk, vec in zip(chunks, vecs):
                batch.add_object(
                    properties={
                        "chunk_id": chunk["id"],
                        "cite_id": chunk["cite_id"],
                        "docid": chunk["docid"],
                        "slug": chunk["slug"],
                        "article": chunk["article"],
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "url": chunk["url"],
                        "repealed": bool(chunk.get("repealed", False)),
                        "has_repealed_parts": bool(chunk.get("has_repealed_parts", False)),
                    },
                    vector=list(map(float, vec)),
                )
        if coll.batch.failed_objects:
            print(f"  ⚠️  {len(coll.batch.failed_objects)} failed inserts")
            print("   ", coll.batch.failed_objects[0].message)

        count = coll.aggregate.over_all(total_count=True).total_count
        print(f"  loaded {count} objects into {COLLECTION} (expected {len(chunks)})")
    finally:
        client.close()


def query(text: str, k: int = 5) -> None:
    """Vector search. 'query: ' prefix — the other half of the e5 convention."""
    vec = encoder().encode([f"query: {text}"], normalize_embeddings=True)[0]
    client = connect()
    try:
        coll = client.collections.get(COLLECTION)
        res = coll.query.near_vector(
            near_vector=list(map(float, vec)), limit=k, return_metadata=["distance"]
        )
        print(f'\n"{text}"')
        for i, o in enumerate(res.objects, 1):
            p = o.properties
            print(
                f"  {i}. {p['cite_id']:<26} sim={1 - o.metadata.distance:.3f}  {p['title'][:52]}"
            )
    finally:
        client.close()


if __name__ == "__main__":
    if "--query" in sys.argv:
        query(sys.argv[sys.argv.index("--query") + 1])
    else:
        build()
