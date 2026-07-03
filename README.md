# PixelRAG

Visual document search — RAG over document **images** instead of extracted text. PDF pages are rendered to image tiles, embedded with a vision-language model, and searched visually. No OCR, no text extraction, no chunking: tables, charts, diagrams, and layout survive intact because the model *sees* the page.

## How it works

```
PDF ──render──▶ page tiles (PNG) ──embed──▶ vectors ──▶ FAISS / Qdrant
                                                            │
query text ──embed──▶ vector search ──▶ top-K tiles ──rerank──▶ top-N
                                                            │
                                          VLM reads tiles ──▶ answer
```

| Stage | Model | Notes |
|---|---|---|
| Embedding | `nvidia/llama-nemotron-embed-vl-1b-v2` | Single vector per page (~1.4B params, runs on CPU) |
| Reranking | `nvidia/llama-nemotron-rerank-vl-1b-v2` | Cross-encoder over [query, image] pairs |
| Answering | Qwen2.5-VL (or any OpenAI-compatible VLM) | Reads the top tiles, answers the question |

**Why single-vector instead of late interaction (ColPali-style):** late interaction stores ~773 vectors per page — ~185 GB for 50K pages. A single vector is ~200 MB for the same corpus, and the cross-encoder reranker recovers the accuracy gap (see the Nemotron paper, Table 6).

## Repository layout

Two ways to use it:

### 1. Standalone CLI pipeline (FAISS, local files)

```bash
python download_models.py            # one-time, ~3 GB per model
python render.py path/to/report.pdf  # PDF pages -> tiles/*.png (150 DPI, pymupdf)
python index.py                      # tiles -> FAISS index in ./index/ (checkpointed, resumable)
python query.py "What was Q3 revenue?"
```

`query.py` sends the top tiles to a hosted VLM. Configure the provider in `.env`:

```env
PROVIDER=openrouter        # gemini | huggingface | groq | openrouter | nvidia
OPENROUTER_API_KEY=...     # key env var for the chosen provider
# MODEL=...                # optional override of the provider default
```

### 2. FastAPI service (Qdrant, multi-user)

`app.py` — a persistent service with per-user isolation, designed to sit behind an authenticated gateway.

```bash
uvicorn app:app --host 0.0.0.0 --port 8100 --workers 1
# always 1 worker: models are ~4 GB on GPU and must not be duplicated
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model/Qdrant status, queue depth |
| `POST /index` | Accepts a base64 tile, returns **202** immediately; embedding + upsert happen in a background queue |
| `POST /query` | Embed query → Qdrant search (filtered by `user_id`) → batched rerank → optional VLM answer |
| `DELETE /file/{file_id}` | Removes a document's vectors and tiles |

Configuration via environment / `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance |
| `PIXELRAG_VISUAL_COLLECTION` | `visual_embeddings` | Collection name |
| `PIXELRAG_RETRIEVE_K` | `20` | Candidates fetched from Qdrant |
| `PIXELRAG_RERANK_BATCH_SIZE` | `8` | Images per reranker pass (prevents GPU OOM) |
| `PIXELRAG_RERANK_TOP_N` | `3` | Final results returned |
| `PIXELRAG_MAX_IMAGE_MB` | `30` | Reject larger uploads |
| `PIXELRAG_EMBED_QUEUE_MAX` | `500` | Background queue backpressure limit |
| `VLLM_QWEN_URL` | *(empty)* | OpenAI-compatible VLM endpoint; leave empty to skip the answer step |
| `TILES_DIR` | `tiles` | Where page tiles are stored |

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+. GPU is optional — everything falls back to CPU (float32).

## Security notes

- Index/delete IDs are validated (`[A-Za-z0-9_-]{1,128}`) to prevent path traversal.
- The service itself has **no authentication** — `user_id` is caller-supplied. Run it on a private network behind a gateway that enforces identity; do not expose the port publicly.
- API keys are read from environment variables only; nothing is stored in the repo.
