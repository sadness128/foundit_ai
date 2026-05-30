import json
import os
import sqlite3
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
        {"found_id": item["id"], "score": item["score"]}
        for item in normalize_top5(top5)
    ]


def add_item(col, item_id: str, image_url: str, image_hash: str, embedding: list[float]) -> None:
    col.add(
        ids=[item_id],
        embeddings=[embedding],
        metadatas=[{"image_url": image_url, "image_hash": image_hash}],
    )


def exists_by_hash(col, image_hash: str) -> bool:
    if col.count() == 0:
        return False
    result = col.get(where={"image_hash": image_hash}, include=["metadatas"])
    return bool(result.get("ids"))


def search_top5_in_found(lost_embedding: list[float]) -> list[dict[str, Any]]:
    if found_col.count() == 0:
        return []

    result = found_col.query(
        query_embeddings=[lost_embedding],
        n_results=min(TOP_K, found_col.count()),
        include=["distances", "metadatas"],
    )

    ids = result["ids"][0]
    distances = result["distances"][0]
    metas = result["metadatas"][0]

    return [
        {
            "id": item_id,
            "score": round(1 - float(distance), 4),
            "image_url": meta.get("image_url", ""),
        }
        for item_id, distance, meta in zip(ids, distances, metas)
    ]


def normalize_top5(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup = {}

    for item in items:
        found_id = item["id"]
        score = round(float(item["score"]), 4)

        if found_id not in dedup or score > dedup[found_id]["score"]:
            dedup[found_id] = {
                "id": found_id,
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
