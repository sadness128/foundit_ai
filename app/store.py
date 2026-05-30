import json
import os
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Any

import chromadb

TOP_K = 5
CHROMA_PATH = os.getenv("FOUNDIT_CHROMA_PATH", "./chroma_db")
CACHE_DB = os.getenv("FOUNDIT_CACHE_DB", "./top5_cache.sqlite3")
COSINE_COLLECTION_METADATA = {"hnsw:space": "cosine"}

client = chromadb.PersistentClient(path=CHROMA_PATH)
lost_col = client.get_or_create_collection(
    "lost_items",
    metadata=COSINE_COLLECTION_METADATA,
)
found_col = client.get_or_create_collection(
    "found_items",
    metadata=COSINE_COLLECTION_METADATA,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect():
    conn = sqlite3.connect(CACHE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lost_top5 (
                lost_id TEXT PRIMARY KEY,
                top5_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


init_cache()


def to_match_response(top5: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"found_id": item["found_id"], "score": item["score"]}
        for item in normalize_top5(top5)
    ]


def image_record_id(prefix: str, item_id: int, image_url: str) -> str:
    url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{item_id}_{url_hash}"


def add_item_image(
    col,
    prefix: str,
    item_id: int,
    image_url: str,
    image_hash: str,
    embedding: list[float],
) -> None:
    col.upsert(
        ids=[image_record_id(prefix, item_id, image_url)],
        embeddings=[embedding],
        metadatas=[
            {
                "item_id": item_id,
                "image_url": image_url,
                "image_hash": image_hash,
            }
        ],
    )


def get_item_embeddings(col, item_id: int) -> list[list[float]]:
    if col.count() == 0:
        return []
    result = col.get(where={"item_id": item_id}, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None:
        return []
    return embeddings


def get_all_item_embeddings(col) -> tuple[list[int], list[list[float]]]:
    if col.count() == 0:
        return [], []

    result = col.get(include=["embeddings", "metadatas"])
    item_ids = []
    embeddings = []

    for metadata, embedding in zip(result.get("metadatas", []), result.get("embeddings", [])):
        item_id = metadata.get("item_id") if metadata else None
        if item_id is None:
            continue
        item_ids.append(int(item_id))
        embeddings.append(embedding)

    return item_ids, embeddings


def search_top5_in_found(lost_embeddings: list[list[float]]) -> list[dict[str, Any]]:
    if found_col.count() == 0:
        return []
    if len(lost_embeddings) == 0:
        return []

    result = found_col.query(
        query_embeddings=lost_embeddings,
        n_results=found_col.count(),
        include=["distances", "metadatas"],
    )

    candidates = []
    for distances, metas in zip(result["distances"], result["metadatas"]):
        for distance, meta in zip(distances, metas):
            found_id = meta.get("item_id") if meta else None
            if found_id is None:
                continue
            candidates.append(
                {
                    "found_id": int(found_id),
                    "score": round(1 - float(distance), 4),
                    "image_url": meta.get("image_url", ""),
                }
            )

    return normalize_top5(candidates)


def search_top5_for_lost(lost_id: int) -> list[dict[str, Any]]:
    return search_top5_in_found(get_item_embeddings(lost_col, lost_id))


def normalize_top5(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup = {}

    for item in items:
        found_id = int(item["found_id"])
        score = round(float(item["score"]), 4)

        if found_id not in dedup or score > dedup[found_id]["score"]:
            dedup[found_id] = {
                "found_id": found_id,
                "score": score,
                "image_url": item.get("image_url", ""),
            }

    return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:TOP_K]


def get_cache(lost_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT top5_json FROM lost_top5 WHERE lost_id = ?",
            (lost_id,),
        ).fetchone()

    return json.loads(row["top5_json"]) if row else []


def set_cache(lost_id: str, top5: list[dict[str, Any]]) -> None:
    payload = json.dumps(normalize_top5(top5), ensure_ascii=False)

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO lost_top5 (lost_id, top5_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(lost_id) DO UPDATE SET
                top5_json = excluded.top5_json,
                updated_at = excluded.updated_at
            """,
            (lost_id, payload, now()),
        )


def get_all_cache() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT lost_id, top5_json, updated_at FROM lost_top5 ORDER BY lost_id"
        ).fetchall()

    return [
        {
            "lost_id": row["lost_id"],
            "top5": json.loads(row["top5_json"]),
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def min_score(top5: list[dict[str, Any]]) -> float:
    if not top5:
        return float("-inf")
    return min(float(item["score"]) for item in top5)
