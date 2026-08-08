"""Embed data/chunks.jsonl and load it into Weaviate.

Vectors are computed client-side (D3: BYO vectors keep the model swappable) with
the model in OPENLAW_EMBED_MODEL. If data/embeddings/ holds precomputed vectors —
e.g. produced on a GPU box — they are used instead of encoding locally.

⚠️ This is an e5-family model: the "passage: " / "query: " prefixes are mandatory.
Dropping them costs a lot of retrieval accuracy, silently.

Indexes are versioned and every build gets an explicit name — nothing is ever
overwritten, and the `Article` alias decides which version queries actually hit.

Usage:
    uv run python index.py --name all_codes        # build + activate
    uv run python index.py --name x --no-activate  # build, keep serving the old one
    uv run python index.py --versions              # list, * marks active
    uv run python index.py --activate labor_only   # atomic switch / rollback
    uv run python index.py --drop old_name
    uv run python index.py --query "..."           # retrieval smoke test
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

from config import get

CHUNKS = Path(get("paths.chunks"))
EMB_DIR = Path(get("paths.embeddings"))
MANIFEST = Path(get("paths.index_manifest"))

# Queries resolve through the alias; builds write to a new versioned collection.
COLLECTION = get("weaviate.alias", env="OPENLAW_COLLECTION")
PREFIX = get("weaviate.collection_prefix")

MODEL = get("embedding.model", env="OPENLAW_EMBED_MODEL")
QUERY_PREFIX = get("embedding.query_prefix")
PASSAGE_PREFIX = get("embedding.passage_prefix")
DIM = get("embedding.dim")

HOST = get("weaviate.host", env="OPENLAW_WEAVIATE_HOST")
PORT = get("weaviate.port", env="OPENLAW_WEAVIATE_PORT")

SEARCH_MODE = get("retrieval.mode", env="OPENLAW_SEARCH_MODE")
HYBRID_ALPHA = get("retrieval.alpha", env="OPENLAW_HYBRID_ALPHA")
TOP_K = get("retrieval.top_k")
TITLE_BOOST = get("retrieval.title_boost")
BM25_PROPS = [f"title^{TITLE_BOOST}", "text"]


def passage(chunk: dict) -> str:
    """Exactly what gets embedded. Must match any external embedding script byte for byte."""
    return f"{PASSAGE_PREFIX}{chunk['title']}\n{chunk['text']}"


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line]


@lru_cache(maxsize=1)
def encoder():
    """Cached: without this the 2.2GB model was reloaded on every single query."""
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


def versions(client) -> list[str]:
    """Existing index versions. Names are chosen explicitly, never generated."""
    return sorted(n for n in client.collections.list_all() if n.startswith(PREFIX))


def active_version(client) -> str | None:
    if not client.alias.exists(alias_name=COLLECTION):
        return None
    return client.alias.get(alias_name=COLLECTION).collection


def activate(client, name: str) -> None:
    """Point the alias at `name`. Atomic: queries switch over with no downtime."""
    if client.alias.exists(alias_name=COLLECTION):
        client.alias.update(alias_name=COLLECTION, new_target_collection=name)
    else:
        client.alias.create(alias_name=COLLECTION, target_collection=name)


def record(name: str, chunks: list[dict], activated: bool) -> None:
    """Append this build to the manifest, so a version's contents stay explainable."""
    from collections import Counter

    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    entries = [e for e in entries if e["version"] != name]
    entries.append(
        {
            "version": name,
            "chunks": len(chunks),
            "articles": len({c["cite_id"] for c in chunks if c["article"]}),
            "documents": dict(Counter(c["slug"] for c in chunks)),
            "embed_model": MODEL,
            "activated": activated,
        }
    )
    MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


def build(name: str, activate_new: bool = True) -> None:
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances

    chunks = load_chunks()
    print(f"  {len(chunks)} chunks from {CHUNKS}")
    vecs = vectors_for(chunks)

    client = connect()
    try:
        existing = versions(client)
        name = name if name.startswith(PREFIX) else f"{PREFIX}{name}"
        if name in existing:
            sys.exit(f"{name} already exists — drop it first or choose another name")
        print(f"  building {name} (existing: {', '.join(existing) or 'none'})")

        client.collections.create(
            name,
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

        coll = client.collections.get(name)
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
        print(f"  loaded {count} objects into {name} (expected {len(chunks)})")

        # Only repoint the alias once the new index is known good — a half-loaded
        # index must never become the one the demo queries.
        ok = count == len(chunks) and not coll.batch.failed_objects
        if activate_new and ok:
            previous = active_version(client)
            activate(client, name)
            print(f"  alias {COLLECTION} -> {name}" + (f" (was {previous})" if previous else ""))
        elif not ok:
            print(f"  ⚠️  {name} incomplete — alias left pointing at {active_version(client)}")
        record(name, chunks, activated=bool(activate_new and ok))
    finally:
        client.close()


def show_versions() -> None:
    client = connect()
    try:
        current = active_version(client)
        manifest = {e["version"]: e for e in (json.loads(MANIFEST.read_text()) if MANIFEST.exists() else [])}
        for v in versions(client):
            n = client.collections.get(v).aggregate.over_all(total_count=True).total_count
            docs = manifest.get(v, {}).get("documents", {})
            mark = "* active" if v == current else "        "
            print(f"  {mark}  {v:<12} {n:>6} chunks  {len(docs)} docs: {', '.join(sorted(docs))[:70]}")
        if not versions(client):
            print("  no index versions yet — run: uv run python index.py")
    finally:
        client.close()


def score_of(obj) -> float:
    """near_vector reports distance, bm25/hybrid report score — normalise to one number."""
    meta = obj.metadata
    if getattr(meta, "distance", None) is not None:
        return 1 - meta.distance
    return getattr(meta, "score", None) or 0.0


def search(coll, question: str, mode: str | None = None, alpha: float | None = None, k: int = TOP_K):
    """One entry point for all three retrieval modes.

    Note the query text differs per leg: BM25 gets the bare question, while the
    vector leg gets the 'query: ' prefix the e5 convention requires. Hybrid needs
    both, which is why the prefix is applied here and not by the caller.
    """
    mode = mode or SEARCH_MODE
    alpha = HYBRID_ALPHA if alpha is None else alpha

    if mode == "bm25":
        return coll.query.bm25(
            query=question, query_properties=BM25_PROPS, limit=k, return_metadata=["score"]
        ).objects

    vec = list(
        map(float, encoder().encode([f"{QUERY_PREFIX}{question}"], normalize_embeddings=True)[0])
    )
    if mode == "vector":
        return coll.query.near_vector(
            near_vector=vec, limit=k, return_metadata=["distance"]
        ).objects
    return coll.query.hybrid(
        query=question,
        vector=vec,
        alpha=alpha,
        query_properties=BM25_PROPS,
        limit=k,
        return_metadata=["score"],
    ).objects


def query(text: str, k: int = 5, mode: str | None = None) -> None:
    client = connect()
    try:
        coll = client.collections.get(COLLECTION)
        objs = search(coll, text, mode=mode, k=k)
        print(f'\n"{text}"  [{mode or SEARCH_MODE}]')
        for i, o in enumerate(objs, 1):
            print(f"  {i}. {o.properties['cite_id']:<26} {score_of(o):.3f}  {o.properties['title'][:52]}")
    finally:
        client.close()


def _flag(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    if "--query" in sys.argv:
        query(_flag("--query"), k=int(_flag("--k", 5)), mode=_flag("--mode"))
    elif "--versions" in sys.argv:
        show_versions()
    elif "--activate" in sys.argv:
        target = _flag("--activate")
        target = target if target.startswith(PREFIX) else f"{PREFIX}{target}"
        c = connect()
        try:
            if target not in versions(c):
                sys.exit(f"{target} does not exist. Known: {', '.join(versions(c))}")
            activate(c, target)
            print(f"  alias {COLLECTION} -> {target}")
        finally:
            c.close()
    elif "--drop" in sys.argv:
        target = _flag("--drop")
        target = target if target.startswith(PREFIX) else f"{PREFIX}{target}"
        c = connect()
        try:
            if target == active_version(c):
                sys.exit(f"{target} is active — activate another version before dropping it")
            c.collections.delete(target)
            print(f"  dropped {target}")
        finally:
            c.close()
    else:
        name = _flag("--name")
        if not name:
            sys.exit(
                "Index names are explicit — pass one:\n"
                "  uv run python index.py --name all_codes\n"
                "  uv run python index.py --name all_codes --no-activate\n"
                "  uv run python index.py --versions | --activate NAME | --drop NAME"
            )
        build(name, activate_new="--no-activate" not in sys.argv)
