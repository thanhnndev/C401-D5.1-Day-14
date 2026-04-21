"""
chunking.py - Document chunking module for product knowledge base.

Reads markdown files from data/products_cleaned, extracts metadata from
YAML frontmatter, then uses LangChain MarkdownHeaderTextSplitter to split
into semantic chunks. Outputs data/chunks.jsonl for Qdrant ingestion.
"""
import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter

PRODUCTS_DIR = Path("data/products_cleaned")
CHUNKS_OUTPUT = Path("data/chunks.jsonl")

# Headers that define chunk boundaries (in order of depth)
HEADERS_TO_SPLIT = [
    ("#", "title"),
    ("##", "heading"),
    ("###", "subheading"),
]

# Minimum characters a chunk must have to be kept
MIN_CHUNK_CHARS = 50


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text)."""
    if not raw.startswith("---"):
        return {}, raw

    end = raw.find("---", 3)
    if end == -1:
        return {}, raw

    fm_block = raw[3:end].strip()
    body = raw[end + 3:].strip()

    try:
        meta = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        meta = {}

    return meta, body


def _make_chunk_id(brand: str, product: str, language: str, index: int) -> str:
    """Create a stable, deterministic chunk_id."""
    return f"{brand}__{product}__{language}__{index:03d}"


def chunk_file(path: Path) -> list[dict]:
    """
    Parse a single markdown product file into chunks with full metadata.

    Returns a list of chunk dicts ready for Qdrant ingestion and ground truth mapping.
    """
    raw = path.read_text(encoding="utf-8").strip()
    frontmatter, body = _parse_frontmatter(raw)

    # Core metadata from YAML header (fallback to path-based inference)
    brand: str = frontmatter.get("brand") or path.parts[-3]
    product: str = frontmatter.get("product") or path.stem
    category: str = frontmatter.get("category") or "unknown"
    language: str = frontmatter.get("language") or path.parts[-2]
    order_number: Optional[str] = frontmatter.get("order_number") or None
    source_file: str = frontmatter.get("original_file") or str(
        path.relative_to(PRODUCTS_DIR)
    )

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    lc_docs = splitter.split_text(body)

    chunks: list[dict] = []
    for idx, doc in enumerate(lc_docs):
        text = doc.page_content.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue

        heading = (
            doc.metadata.get("subheading")
            or doc.metadata.get("heading")
            or doc.metadata.get("title")
            or "Overview"
        )

        chunk_id = _make_chunk_id(brand, product, language, idx)

        metadata: dict[str, Any] = {
            "chunk_id": chunk_id,
            "brand": brand,
            "product": product,
            "category": category,
            "language": language,
            "heading": heading,
            "source_file": source_file,
            "char_count": len(text),
            "token_estimate": len(text) // 4,
        }

        # Melag-specific field
        if order_number:
            metadata["order_number"] = order_number

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
            }
        )

    return chunks


def chunk_all(products_dir: Path = PRODUCTS_DIR) -> list[dict]:
    """Walk all markdown files and return the complete list of chunks."""
    all_chunks: list[dict] = []
    for md_file in sorted(products_dir.rglob("*.md")):
        try:
            file_chunks = chunk_file(md_file)
            all_chunks.extend(file_chunks)
        except Exception as exc:
            print(f"[WARN] Skipping {md_file}: {exc}")
    return all_chunks


def save_chunks(chunks: list[dict], output: Path = CHUNKS_OUTPUT) -> None:
    """Write chunks to JSONL file for Qdrant ingestion."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(chunks)} chunks → {output}")


if __name__ == "__main__":
    chunks = chunk_all()
    save_chunks(chunks)

    # Quick stats
    from collections import Counter
    brands = Counter(c["metadata"]["brand"] for c in chunks)
    langs = Counter(c["metadata"]["language"] for c in chunks)
    print(f"[Stats] By brand: {dict(brands)}")
    print(f"[Stats] By language: {dict(langs)}")
    print(f"[Stats] Total chunks: {len(chunks)}")
