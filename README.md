# Foundit AI Similarity Matcher

이미지 임베딩 기반 분실물/습득물 top-5 매칭 서버입니다.

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

## 환경변수

```text
FOUNDIT_PUBLIC_BASE_URL=http://localhost:8000
FOUNDIT_UPLOAD_DIR=./uploaded_images
FOUNDIT_CHROMA_PATH=./chroma_db
FOUNDIT_CACHE_DB=./top5_cache.sqlite3
```

## API

```text
POST   /upload
POST   /lost
POST   /found
GET    /matches/{lost_id}
```

`/lost`, `/found` 요청의 `item_id`는 선택값입니다. 백엔드 DB의 item id를 넣으면 AI 서버는 그 값을 그대로 `lost_id`, `found_id`로 저장하고 응답합니다. 생략하면 AI 서버가 `lost_<uuid>`, `found_<uuid>` 형태의 id를 생성합니다.

## 핵심 동작

- 분실물 등록: 기존 습득물 DB에서 top5 검색 후 cache 저장
- 습득물 등록: 새 습득물과 모든 분실물 embedding만 비교해서 top5 cache 갱신

## 요청 예시

권장 요청:

```json
{
  "item_id": "123",
  "image_url": "http://localhost:8000/images/example.jpg"
}
```

하위 호환 요청:

```json
{
  "image_url": "http://localhost:8000/images/example.jpg"
}
```
