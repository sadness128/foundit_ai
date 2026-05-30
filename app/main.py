import hashlib
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

import app.store as store
from app.model import embed_image

from pydantic import BaseModel


class LostRequest(BaseModel):
    image_url: list[str]
    lost_id: int


class FoundRequest(BaseModel):
    image_url: list[str]
    found_id: int


UPLOAD_DIR = Path(os.getenv("FOUNDIT_UPLOAD_DIR", "./uploaded_images"))
UPLOAD_DIR.mkdir(exist_ok=True)

BASE_URL = os.getenv("FOUNDIT_PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

app = FastAPI(title="Foundit AI Similarity Matcher")
app.mount("/images", StaticFiles(directory=str(UPLOAD_DIR)), name="images")


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")


def load_image_from_url(url: str) -> tuple[Image.Image, str]:
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지를 가져올 수 없습니다: {e}")

    data = res.content
    return open_image(data), image_hash(data)


def build_image_url(filename: str) -> str:
    return f"{BASE_URL}/images/{filename}"


def validate_image_urls(image_urls: list[str]) -> None:
    if not image_urls:
        raise HTTPException(status_code=400, detail="image_url은 비어 있을 수 없습니다.")
    if any(not url or not url.strip() for url in image_urls):
        raise HTTPException(status_code=400, detail="image_url에는 빈 문자열을 넣을 수 없습니다.")


def embed_image_urls(image_urls: Iterable[str]) -> list[tuple[str, str, list[float]]]:
    embedded = []
    for image_url in image_urls:
        image, h = load_image_from_url(image_url)
        embedded.append((image_url, h, embed_image(image)))
    return embedded


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    open_image(data)

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / filename
    path.write_bytes(data)

    return {"image_url": build_image_url(filename)}


@app.post("/api/ai/lost")
def register_lost(req: LostRequest):
    started_at = time.perf_counter()
    validate_image_urls(req.image_url)
    for image_url, h, embedding in embed_image_urls(req.image_url):
        store.add_item_image(store.lost_col, "lost", req.lost_id, image_url, h, embedding)

    top5 = store.search_top5_for_lost(req.lost_id)
    store.set_cache(req.lost_id, top5)

    response_body = {
        "lost_id": req.lost_id,
        "matches": store.to_match_response(top5),
    }
    print(f"[POST /api/ai/lost] response={response_body} elapsed={time.perf_counter() - started_at:.3f}s")
    return response_body


@app.post("/api/ai/found")
def register_found(req: FoundRequest):
    started_at = time.perf_counter()
    validate_image_urls(req.image_url)
    for image_url, h, embedding in embed_image_urls(req.image_url):
        store.add_item_image(store.found_col, "found", req.found_id, image_url, h, embedding)

    lost_ids, _ = store.get_all_item_embeddings(store.lost_col)
    updated = []

    for lost_id in sorted(set(lost_ids)):
        top5 = store.search_top5_for_lost(lost_id)
        store.set_cache(lost_id, top5)
        if any(item["found_id"] == req.found_id for item in top5):
            updated.append(
                {
                    "lost_id": lost_id,
                    "current_top5": store.to_match_response(top5),
                }
            )

    response_body = {
        "registered_found_id": req.found_id,
        "updated_lost_items": updated,
    }
    print(f"[POST /api/ai/found] response={response_body} elapsed={time.perf_counter() - started_at:.3f}s")
    return response_body


@app.get("/api/ai/matches/{lost_id}")
def match_for_lost(lost_id: int):
    lost_embeddings = store.get_item_embeddings(store.lost_col, lost_id)
    if len(lost_embeddings) == 0:
        raise HTTPException(status_code=404, detail="존재하지 않는 분실물입니다.")

    top5 = store.search_top5_in_found(lost_embeddings)
    store.set_cache(lost_id, top5)

    return {
        "lost_id": lost_id,
        "matches": store.to_match_response(top5),
    }
