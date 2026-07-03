"""
PixelRAG FastAPI service — multi-user visual document search.

Pipeline per query:
  1. Embed query text (Nemotron VL 1B, GPU-serialised)
  2. Search Qdrant visual collection filtered by user_id
  3. Batch-rerank candidates in PIXELRAG_RERANK_BATCH_SIZE-image passes (avoids OOM)
  4. Call internal Qwen vLLM with the top image for a text answer

Indexing is fire-and-return (202): the tile is saved to disk immediately and
embedding/Qdrant upsert happens in a background asyncio task — callers are never
blocked by model inference time.

Start:  uvicorn app:app --host 0.0.0.0 --port 8100 --workers 1
        (always 1 worker — models are 4 GB on GPU and must not be duplicated)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from qdrant_client import AsyncQdrantClient, models as qmodels
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor
from openai import AsyncOpenAI


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"

    QDRANT_URL:                  str   = "http://localhost:6333"
    PIXELRAG_VISUAL_COLLECTION:  str   = "visual_embeddings"
    PIXELRAG_EMBED_DIM:          int   = 2048
    PIXELRAG_RETRIEVE_K:         int   = 20    # Qdrant candidates (dev=20, prod=50)
    PIXELRAG_RERANK_BATCH_SIZE:  int   = 8     # images per reranker pass (dev=8, prod=50)
    PIXELRAG_RERANK_TOP_N:       int   = 3     # final results returned
    PIXELRAG_QWEN_CONCURRENCY:   int   = 2     # max parallel Qwen VLM calls
    PIXELRAG_MAX_IMAGE_MB:       int   = 30    # reject images larger than this
    PIXELRAG_EMBED_QUEUE_MAX:    int   = 500   # background queue maxsize (backpressure)
    VLLM_QWEN_URL:               str   = ""
    VLLM_QWEN_MODEL:             str   = "Qwen/Qwen2.5-VL-7B-Instruct"
    TILES_DIR:                   str   = "tiles"

cfg = Settings()

EMBEDDER_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
RERANKER_ID = "nvidia/llama-nemotron-rerank-vl-1b-v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("pixelrag")


# ── Runtime state ─────────────────────────────────────────────────────────────

class _State:
    em_proc:       object             = None
    embedder:      object             = None
    re_proc:       object             = None
    reranker:      object             = None
    qdrant:        AsyncQdrantClient  = None
    qwen:          AsyncOpenAI        = None
    gpu_sem:       asyncio.Semaphore  = None
    embed_q:       asyncio.Queue      = None
    _bg_task:      asyncio.Task       = None
    startup_ready: bool               = False   # set True only after all models loaded

_st = _State()


# ── Model loading (called once at startup in threadpool) ──────────────────────

def _load_models() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    log.info(f"Loading embedder {EMBEDDER_ID} on {device}...")
    _st.em_proc  = AutoProcessor.from_pretrained(EMBEDDER_ID, trust_remote_code=True)
    _st.embedder = AutoModel.from_pretrained(
        EMBEDDER_ID, trust_remote_code=True, dtype=dtype, device_map="auto",
    ).eval()

    log.info(f"Loading reranker {RERANKER_ID} on {device}...")
    _st.re_proc = AutoProcessor.from_pretrained(RERANKER_ID, trust_remote_code=True)
    try:
        _st.reranker = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_ID, trust_remote_code=True, dtype=dtype, device_map="auto",
        ).eval()
    except Exception:
        _st.reranker = AutoModel.from_pretrained(
            RERANKER_ID, trust_remote_code=True, dtype=dtype, device_map="auto",
        ).eval()

    log.info("Models loaded successfully")


# ── Qdrant collection setup ───────────────────────────────────────────────────

async def _ensure_collection() -> None:
    exists = await _st.qdrant.collection_exists(cfg.PIXELRAG_VISUAL_COLLECTION)
    if not exists:
        await _st.qdrant.create_collection(
            collection_name=cfg.PIXELRAG_VISUAL_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=cfg.PIXELRAG_EMBED_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )
        await _st.qdrant.create_payload_index(
            cfg.PIXELRAG_VISUAL_COLLECTION, "user_id", qmodels.PayloadSchemaType.KEYWORD,
        )
        await _st.qdrant.create_payload_index(
            cfg.PIXELRAG_VISUAL_COLLECTION, "file_id", qmodels.PayloadSchemaType.KEYWORD,
        )
        log.info(f"Created Qdrant collection '{cfg.PIXELRAG_VISUAL_COLLECTION}' (dim={cfg.PIXELRAG_EMBED_DIM})")
    else:
        info  = await _st.qdrant.get_collection(cfg.PIXELRAG_VISUAL_COLLECTION)
        count = info.points_count or 0
        log.info(f"Loaded Qdrant collection '{cfg.PIXELRAG_VISUAL_COLLECTION}' ({count} points)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _point_id(file_id: str, image_id: str) -> str:
    """Stable UUID-formatted point ID derived from (file_id, image_id)."""
    digest = hashlib.md5(f"{file_id}:{image_id}".encode()).hexdigest()
    return str(uuid.UUID(digest))


def _rel_path(abs_path: Path) -> str:
    """Store tile paths relative to TILES_DIR so they survive CWD changes."""
    try:
        return str(abs_path.resolve().relative_to(Path(cfg.TILES_DIR).resolve()))
    except ValueError:
        return str(abs_path)


def _abs_path(rel: str) -> Path:
    """Resolve a relative tile path back to an absolute path."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return Path(cfg.TILES_DIR).resolve() / p


def _require_ready() -> None:
    if not _st.startup_ready:
        raise HTTPException(status_code=503, detail="PixelRAG models still loading — try again in a moment")


# ── GPU inference helpers (run in threadpool, called inside gpu_sem) ──────────

def _embed_image_sync(img: Image.Image) -> list[float]:
    inputs = _st.em_proc(
        images=[img],
        text=["Represent this document page for retrieval:"],
        return_tensors="pt", padding=True,
    )
    device = next(_st.embedder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _st.embedder(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb[0].cpu().tolist()


def _embed_query_sync(query: str) -> list[float]:
    inputs = _st.em_proc(
        text=[f"Query: {query}"],
        return_tensors="pt", padding=True,
    )
    device = next(_st.embedder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _st.embedder(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb[0].cpu().tolist()


def _batch_rerank_sync(query: str, tile_paths: list[str]) -> list[float]:
    """Rerank tile_paths against query in PIXELRAG_RERANK_BATCH_SIZE-image micro-batches.

    Batching prevents GPU OOM when PIXELRAG_RETRIEVE_K is large (e.g. 50 in prod).
    Each micro-batch is one forward pass; all results are concatenated in order.
    """
    device   = next(_st.reranker.parameters()).device
    all_scores: list[float] = []
    batch_sz = max(1, cfg.PIXELRAG_RERANK_BATCH_SIZE)

    for i in range(0, len(tile_paths), batch_sz):
        batch_paths = tile_paths[i : i + batch_sz]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = _st.re_proc(
            images=images,
            text=[query] * len(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = _st.reranker(**inputs, output_hidden_states=True)

        if hasattr(out, "logits"):
            logits = out.logits
            if logits.dim() == 1:
                batch_scores = logits.tolist()
            else:
                batch_scores = logits[:, -1].tolist()
        else:
            batch_scores = out.hidden_states[-1].mean(dim=-1).mean(dim=-1).tolist()

        all_scores.extend(batch_scores)

    return all_scores


# ── Background embed worker ───────────────────────────────────────────────────

async def _bg_embed_worker() -> None:
    log.info("Background embed worker started")
    while True:
        try:
            task = await _st.embed_q.get()
            if task is None:
                break
            user_id, file_id, image_id, tile_path = task

            img = Image.open(_abs_path(tile_path)).convert("RGB")

            async with _st.gpu_sem:
                vector = await run_in_threadpool(_embed_image_sync, img)

            await _st.qdrant.upsert(
                collection_name=cfg.PIXELRAG_VISUAL_COLLECTION,
                points=[qmodels.PointStruct(
                    id=_point_id(file_id, image_id),
                    vector=vector,
                    payload={
                        "user_id":   user_id,
                        "file_id":   file_id,
                        "image_id":  image_id,
                        "tile_path": tile_path,  # relative path
                    },
                )],
            )
            log.info(f"Indexed: user={user_id} file={file_id} image={image_id}")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.warning(f"Background embed error: {exc}")
        finally:
            try:
                _st.embed_q.task_done()
            except Exception:
                pass


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(_load_models)
    except Exception as exc:
        log.error(f"Model loading failed — service will return 503: {exc}")
        yield
        return

    _st.qdrant  = AsyncQdrantClient(url=cfg.QDRANT_URL)
    _st.gpu_sem = asyncio.Semaphore(1)
    _st.embed_q = asyncio.Queue(maxsize=cfg.PIXELRAG_EMBED_QUEUE_MAX)

    if cfg.VLLM_QWEN_URL:
        _st.qwen = AsyncOpenAI(
            base_url=cfg.VLLM_QWEN_URL.rstrip("/") + "/v1",
            api_key="none",
        )

    await _ensure_collection()
    Path(cfg.TILES_DIR).mkdir(parents=True, exist_ok=True)
    _st.startup_ready = True
    _st._bg_task = asyncio.create_task(_bg_embed_worker())

    log.info("PixelRAG service ready")
    yield

    _st.startup_ready = False
    await _st.embed_q.put(None)
    if _st._bg_task:
        try:
            await asyncio.wait_for(_st._bg_task, timeout=30)
        except asyncio.TimeoutError:
            log.warning("Background worker did not finish in 30 s; cancelling")
            _st._bg_task.cancel()
    if _st.qdrant:
        await _st.qdrant.close()
    log.info("PixelRAG service shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PixelRAG", version="1.0.0", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

# IDs become filesystem path components and glob patterns — restrict to safe chars
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class IndexRequest(BaseModel):
    user_id:   str
    file_id:   str
    image_id:  str
    image_b64: str

    @field_validator("user_id", "file_id", "image_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _SAFE_ID.fullmatch(v):
            raise ValueError("IDs must be 1-128 chars of letters, digits, '_' or '-'")
        return v

    @field_validator("image_b64")
    @classmethod
    def _check_size(cls, v: str) -> str:
        # Rough check: base64 is ~4/3 of raw bytes
        raw_mb = len(v) * 3 / 4 / 1_048_576
        if raw_mb > cfg.PIXELRAG_MAX_IMAGE_MB:
            raise ValueError(f"Image exceeds {cfg.PIXELRAG_MAX_IMAGE_MB} MB limit ({raw_mb:.1f} MB)")
        return v


class QueryRequest(BaseModel):
    user_id:  str
    query:    str
    top_k:    int       = 3
    file_ids: list[str] = []
    use_vlm:  bool      = True


class ImageResult(BaseModel):
    file_id:      str
    image_id:     str
    tile_path:    str
    faiss_score:  float
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    results: List[ImageResult]
    answer:  Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    qdrant_ok = False
    point_count = -1
    if _st.qdrant:
        try:
            info = await _st.qdrant.get_collection(cfg.PIXELRAG_VISUAL_COLLECTION)
            point_count = info.points_count or 0
            qdrant_ok = True
        except Exception:
            pass

    return {
        "status":     "ok" if _st.startup_ready else "loading",
        "models":     "loaded" if _st.startup_ready else "not_loaded",
        "qdrant":     "ok" if qdrant_ok else "error",
        "collection": cfg.PIXELRAG_VISUAL_COLLECTION,
        "points":     point_count,
        "queue_size": _st.embed_q.qsize() if _st.embed_q else 0,
    }


@app.post("/index", status_code=202)
async def index_image(req: IndexRequest):
    """Save tile to disk immediately and queue embedding — returns 202."""
    _require_ready()

    try:
        img_bytes = base64.b64decode(req.image_b64)
        Image.open(io.BytesIO(img_bytes)).verify()  # validate, then discard
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    tile_dir  = Path(cfg.TILES_DIR) / req.user_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_abs  = tile_dir / f"{req.file_id}_{req.image_id}.png"
    tile_rel  = _rel_path(tile_abs)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.save(str(tile_abs), "PNG")

    try:
        _st.embed_q.put_nowait((req.user_id, req.file_id, req.image_id, tile_rel))
    except asyncio.QueueFull:
        # Queue is at capacity — tile is saved, embedding will be retried on restart
        # (Qdrant upsert is idempotent, so re-sending on next upload is safe)
        log.warning(f"Embed queue full — tile saved but not queued: file={req.file_id} image={req.image_id}")
        return {"accepted": False, "reason": "embed_queue_full", "queue_size": _st.embed_q.qsize()}

    return {"accepted": True, "queue_size": _st.embed_q.qsize()}


@app.delete("/file/{file_id}")
async def delete_file(file_id: str):
    """Remove all visual vectors and tiles for a document."""
    _require_ready()

    if not _SAFE_ID.fullmatch(file_id):
        raise HTTPException(status_code=400, detail="Invalid file_id")

    await _st.qdrant.delete(
        collection_name=cfg.PIXELRAG_VISUAL_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="file_id",
                    match=qmodels.MatchValue(value=file_id),
                )]
            )
        ),
    )

    tiles_root = Path(cfg.TILES_DIR).resolve()
    deleted = 0
    for tile in tiles_root.rglob(f"{file_id}_*.png"):
        try:
            tile.unlink()
            deleted += 1
        except Exception as exc:
            log.warning(f"Could not delete tile {tile}: {exc}")

    log.info(f"Deleted vectors and {deleted} tiles for file_id={file_id}")
    return {"file_id": file_id, "tiles_deleted": deleted}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Text query → visual search → batched rerank → optional Qwen VLM answer."""
    _require_ready()

    # 1. Embed query text (GPU sem — serialises with background embed work)
    async with _st.gpu_sem:
        query_vec = await run_in_threadpool(_embed_query_sync, req.query)

    # 2. Build Qdrant filter
    must = [qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=req.user_id))]
    if req.file_ids:
        must.append(qmodels.FieldCondition(
            key="file_id",
            match=qmodels.MatchAny(any=req.file_ids),
        ))

    # 3. Qdrant visual search (qdrant-client ≥ 1.10 uses query_points, not search)
    _resp = await _st.qdrant.query_points(
        collection_name=cfg.PIXELRAG_VISUAL_COLLECTION,
        query=query_vec,
        query_filter=qmodels.Filter(must=must),
        limit=min(cfg.PIXELRAG_RETRIEVE_K, max(req.top_k * 10, 10)),
        with_payload=True,
    )
    hits = _resp.points

    if not hits:
        raise HTTPException(status_code=404, detail="No visual results found for this user")

    # 4. Validate tile paths (skip missing files — might have been cleaned up)
    candidates = []
    for h in hits:
        rel = h.payload.get("tile_path", "")
        abs_p = _abs_path(rel)
        if abs_p.exists():
            candidates.append({
                "file_id":     h.payload["file_id"],
                "image_id":    h.payload["image_id"],
                "tile_path":   rel,
                "faiss_score": h.score,
            })

    if not candidates:
        raise HTTPException(status_code=404, detail="Visual results found but tile files are missing")

    # 5. Batched rerank (all batches run inside GPU sem as one logical op)
    tile_paths = [c["tile_path"] for c in candidates]
    async with _st.gpu_sem:
        rerank_scores = await run_in_threadpool(_batch_rerank_sync, req.query, tile_paths)

    for c, score in zip(candidates, rerank_scores):
        c["rerank_score"] = score

    top = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[: req.top_k]

    # 6. Optional Qwen VLM answer from the top-scored tile
    answer = None
    if req.use_vlm and _st.qwen and top:
        answer = await _ask_qwen(req.query, _abs_path(top[0]["tile_path"]))

    return QueryResponse(
        results=[ImageResult(**c) for c in top],
        answer=answer,
    )


async def _ask_qwen(query: str, tile_path: Path) -> Optional[str]:
    try:
        b64 = base64.b64encode(tile_path.read_bytes()).decode()
        response = await _st.qwen.chat.completions.create(
            model=cfg.VLLM_QWEN_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": (
                    "You are a document analyst. Answer the following question "
                    "using only the information visible in this image.\n\n"
                    f"Question: {query}"
                )},
            ]}],
            max_tokens=1024,
            timeout=30,
        )
        return response.choices[0].message.content
    except Exception as exc:
        log.warning(f"Qwen VLM call failed: {exc}")
        return None
