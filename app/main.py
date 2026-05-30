import hashlib
import os
import uuid
from io import BytesIO
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

import app.store as store
from app.model import dot_scores, embed_image

from pydantic import BaseModel


class ImageRequest(BaseModel):
    image_url: str


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


def create_ai_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


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


@app.post("/lost")
def register_lost(req: ImageRequest):
    image_url = req.image_url
    image, h = load_image_from_url(image_url)

    if store.exists_by_hash(store.lost_col, h):
        raise HTTPException(status_code=409, detail="이미 등록된 분실물입니다.")

    lost_id = create_ai_id("lost")
    emb = embed_image(image)

    store.add_item(store.lost_col, lost_id, image_url, h, emb)

    top5 = store.search_top5_in_found(emb)
    store.set_cache(lost_id, top5)

    return {
        "lost_id": lost_id,
        "matches": store.to_match_response(top5),
    }


@app.post("/found")
def register_found(req: ImageRequest):
    image_url = req.image_url
    image, h = load_image_from_url(image_url)

    if store.exists_by_hash(store.found_col, h):
        raise HTTPException(status_code=409, detail="이미 등록된 습득물입니다.")

    found_id = create_ai_id("found")
    found_emb = embed_image(image)

    store.add_item(store.found_col, found_id, image_url, h, found_emb)

    lost_data = store.lost_col.get(include=["embeddings"])
    lost_ids = lost_data.get("ids", [])
    lost_embeddings = lost_data.get("embeddings", [])

    if not lost_ids:
        return {"registered_found_id": found_id, "updated_lost_items": []}

    scores = dot_scores(lost_embeddings, found_emb)
    updated = []

    for lost_id, score in zip(lost_ids, scores):
        score = round(float(score), 4)
        top5 = store.get_cache(lost_id)

        if len(top5) < store.TOP_K or score > store.min_score(top5):
            new_top5 = store.normalize_top5(
                [
                    *top5,
                    {
                        "id": found_id,
                        "score": score,
                        "image_url": image_url,
                    },
                ]
            )
            store.set_cache(lost_id, new_top5)

            if any(item["id"] == found_id for item in new_top5):
                updated.append(
                    {
                        "lost_id": lost_id,
                        "current_top5": store.to_match_response(new_top5),
                    }
                )

    return {
        "registered_found_id": found_id,
        "updated_lost_items": updated,
    }


@app.get("/matches/{lost_id}")
def match_for_lost(lost_id: str):
    result = store.lost_col.get(ids=[lost_id])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="존재하지 않는 분실물입니다.")
    top5 = store.get_cache(lost_id)
    return {
        "lost_id": lost_id,
        "matches": store.to_match_response(top5),
    }
