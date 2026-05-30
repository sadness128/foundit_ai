import threading

import numpy as np
import open_clip
import torch
from PIL import Image

_model = None
_preprocess = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = threading.Lock()


def _load_model():
    global _model, _preprocess

    if _model is not None:
        return _model, _preprocess, _device

    with _lock:
        if _model is None:
            print("CLIP 모델 로딩 중...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "MobileCLIP-S2",
                pretrained="datacompdr",
            )
            model.to(_device)
            model.eval()

            _model = model
            _preprocess = preprocess
            print(f"CLIP 모델 로딩 완료: {_device}")

    return _model, _preprocess, _device


def embed_image(image: Image.Image) -> list[float]:
    model, preprocess, device = _load_model()
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb.squeeze().cpu().tolist()


def dot_scores(matrix_embeddings, query_embedding) -> np.ndarray:
    """embedding을 normalize해서 저장하므로 cosine similarity = dot product."""
    matrix = np.asarray(matrix_embeddings, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    return matrix @ query
