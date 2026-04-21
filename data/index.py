"""
index.py - Hybrid search indexer: Jina dense + BM25 sparse → Qdrant.

Architecture:
  - Dense vectors : Jina Embeddings API (jina-embeddings-v5-text-small, dim=256)
  - Sparse vectors: fastembed BM25 (Qdrant/bm25) — lexical matching
  - Fusion        : Qdrant RRF (Reciprocal Rank Fusion) at query time

Collection schema:
  vectors        → {"dense": VectorParams(256, COSINE)}
  sparse_vectors → {"bm25": SparseVectorParams(modifier=IDF)}

Usage:
    python data/index.py                  # upsert into existing collection
    python data/index.py --recreate       # drop & recreate collection first
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastembed.sparse.bm25 import Bm25
from qdrant_client import AsyncQdrantClient, models

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_FILE = Path("data/chunks.jsonl")

JINA_API_KEY        = os.environ["EMBEDDING_API_KEY"]
JINA_MODEL          = os.environ["EMBEDDING_MODEL_NAME"]
JINA_EMBED_URL      = "https://api.jina.ai/v1/embeddings"

QDRANT_URL          = os.environ["QRANT_URL"]
QDRANT_API_KEY      = os.environ["QRANT_API_KEY"]
COLLECTION_NAME     = os.getenv("QDRANT_COLLECTION_NAME", "products")

DENSE_DIM           = 256       # jina-embeddings-v5-text-small
EMBED_BATCH_SIZE    = 32
UPSERT_BATCH_SIZE   = 64

BM25_MODEL          = "Qdrant/bm25"   # fastembed sparse model


# ── Dense embedding (Jina API) ────────────────────────────────────────────────

async def _embed_batch(texts: list[str], http: httpx.AsyncClient) -> list[list[float]]:
    resp = await http.post(
        JINA_EMBED_URL,
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": JINA_MODEL, "input": texts, "dimensions": DENSE_DIM},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in sorted(resp.json()["data"], key=lambda x: x["index"])]


async def embed_all(texts: list[str], http: httpx.AsyncClient) -> list[list[float]]:
    all_vecs: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        vecs = await _embed_batch(texts[start : start + EMBED_BATCH_SIZE], http)
        all_vecs.extend(vecs)
        print(f"  [Dense] {min(start + EMBED_BATCH_SIZE, total)}/{total}", end="\r")
    print()
    return all_vecs


# ── Sparse embedding (BM25 via fastembed) ─────────────────────────────────────

def bm25_encode_all(texts: list[str]) -> list[models.SparseVector]:
    """Return BM25 sparse vectors for all texts (runs locally, no API call)."""
    print(f"  [BM25 ] Fitting BM25 model '{BM25_MODEL}'...")
    encoder = Bm25(BM25_MODEL)
    # fastembed returns a generator of SparseEmbedding objects
    sparse_embeddings = list(encoder.embed(texts))
    return [
        models.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in sparse_embeddings
    ]


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

async def ensure_collection(qc: AsyncQdrantClient, recreate: bool) -> None:
    exists = await qc.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        print(f"[INFO] Dropping existing collection '{COLLECTION_NAME}'...")
        await qc.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        print(f"[INFO] Creating hybrid collection '{COLLECTION_NAME}'...")
        await qc.create_collection(
            collection_name=COLLECTION_NAME,
            # Dense vector config
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            # Sparse vector config (BM25)
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # IDF weighting for BM25
                )
            },
        )
        # Payload indexes for filtered search
        for field, schema in [
            ("brand",    models.PayloadSchemaType.KEYWORD),
            ("language", models.PayloadSchemaType.KEYWORD),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("product",  models.PayloadSchemaType.KEYWORD),
        ]:
            await qc.create_payload_index(COLLECTION_NAME, field, schema)
        print(f"[OK] Collection '{COLLECTION_NAME}' created (dense={DENSE_DIM}D + BM25 sparse).")
    else:
        print(f"[INFO] Collection '{COLLECTION_NAME}' exists — upserting.")


async def upsert_chunks(
    qc: AsyncQdrantClient,
    chunks: list[dict],
    dense_vecs: list[list[float]],
    sparse_vecs: list[models.SparseVector],
) -> None:
    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        batch_chunks  = chunks[start:end]
        batch_dense   = dense_vecs[start:end]
        batch_sparse  = sparse_vecs[start:end]

        points = [
            models.PointStruct(
                id=abs(hash(c["chunk_id"])) % (2**63),
                vector={
                    "dense": dv,
                    "bm25":  sv,
                },
                payload={
                    "chunk_id": c["chunk_id"],
                    "text":     c["text"],
                    **c["metadata"],
                },
            )
            for c, dv, sv in zip(batch_chunks, batch_dense, batch_sparse)
        ]
        await qc.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  [Upsert] {min(end, total)}/{total}", end="\r")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(recreate: bool = False) -> None:
    # 1. Load chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run `python data/chunking.py` first.")
    chunks = [json.loads(l) for l in CHUNKS_FILE.read_text().splitlines() if l.strip()]
    texts  = [c["text"] for c in chunks]
    print(f"[INFO] Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # 2. Dense vectors (async Jina API)
    t0 = time.perf_counter()
    print(f"[INFO] Computing dense embeddings via Jina API (batch={EMBED_BATCH_SIZE})...")
    async with httpx.AsyncClient() as http:
        dense_vecs = await embed_all(texts, http)
    print(f"[OK]  Dense embeddings done in {time.perf_counter() - t0:.1f}s")

    # 3. Sparse BM25 vectors (local, fastembed)
    t1 = time.perf_counter()
    print(f"[INFO] Computing BM25 sparse vectors (fastembed '{BM25_MODEL}')...")
    sparse_vecs = bm25_encode_all(texts)
    print(f"[OK]  BM25 done in {time.perf_counter() - t1:.1f}s  "
          f"(avg nnz={sum(len(sv.indices) for sv in sparse_vecs)//len(sparse_vecs)})")

    # 4. Upsert to Qdrant
    qc = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        await ensure_collection(qc, recreate=recreate)

        print(f"[INFO] Upserting {len(chunks)} points (dense + BM25)...")
        t2 = time.perf_counter()
        await upsert_chunks(qc, chunks, dense_vecs, sparse_vecs)
        print(f"[OK]  Upsert done in {time.perf_counter() - t2:.1f}s")

        info = await qc.get_collection(COLLECTION_NAME)
        print(f"\n✅  Hybrid collection '{COLLECTION_NAME}': {info.points_count} points indexed.")
        print(f"    Dense : {DENSE_DIM}D Jina cosine")
        print(f"    Sparse: BM25 (IDF-weighted, Qdrant/bm25)")
        print(f"\n💡  Query example (RRF fusion):")
        print(f"    qc.query_points(collection='{COLLECTION_NAME}',")
        print(f"        prefetch=[")
        print(f"            Prefetch(query=dense_vec,  using='dense',  limit=20),")
        print(f"            Prefetch(query=sparse_vec, using='bm25',   limit=20),")
        print(f"        ],")
        print(f"        query=FusionQuery(fusion=Fusion.RRF), limit=5)")
    finally:
        await qc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid (dense+BM25) indexer for Qdrant.")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the Qdrant collection before indexing.")
    args = parser.parse_args()
    asyncio.run(main(recreate=args.recreate))
