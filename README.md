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
POST   /api/ai/lost
POST   /api/ai/found
GET    /api/ai/matches/{lost_id}
```

`lost_id`, `found_id`, `image_url`은 필수값입니다. `image_url`은 문자열 리스트이며, 빈 리스트는 허용하지 않습니다. AI 서버는 별도 ID를 생성하지 않고 백엔드에서 받은 Long ID를 그대로 저장하고 응답합니다.

## 핵심 동작

- 분실물 등록: 여러 분실물 이미지를 `(lost_id, image_url)` 기준으로 저장한 뒤 기존 습득물 이미지들과 비교해 top5 cache 저장
- 습득물 등록: 여러 습득물 이미지를 `(found_id, image_url)` 기준으로 저장한 뒤 모든 분실물의 top5 cache 갱신
- 매칭 점수: 한 분실물의 모든 이미지와 한 습득물의 모든 이미지 조합 중 가장 높은 유사도 점수를 해당 `found_id`의 최종 score로 사용

## 요청 예시

분실물 등록:

```json
{
  "lost_id": 123,
  "image_url": [
    "http://localhost:8080/images/lost1.jpg",
    "http://localhost:8080/images/lost2.jpg"
  ]
}
```

습득물 등록:

```json
{
  "found_id": 456,
  "image_url": [
    "http://localhost:8080/images/found1.jpg",
    "http://localhost:8080/images/found2.jpg"
  ]
}
```

매칭 결과:

```json
{
  "lost_id": 123,
  "matches": [
    {
      "found_id": 456,
      "score": 0.95
    }
  ]
}
```
