"""
Step 3 — Search the index, rerank, and ask a VLM to answer.
Usage: python query.py "What is the total revenue in Q3?"

Pipeline (from Nemotron paper Table 6 — near late-interaction accuracy, 2000x less storage):
  1. Embed query with nvidia/llama-nemotron-embed-vl-1b-v2
  2. Retrieve top-50 pages via FAISS (fast vector search)
  3. Rerank top-50 with nvidia/llama-nemotron-rerank-vl-1b-v2 (cross-encoder)
  4. Send top-3 tiles to VLM reader for a multi-image answer

Configure PROVIDER and MODEL in .env.
"""

import os
import sys
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import faiss
from transformers import AutoProcessor, AutoModel, AutoModelForSequenceClassification
from openai import OpenAI

INDEX_DIR    = Path("index")
EMBEDDER_ID  = "nvidia/llama-nemotron-embed-vl-1b-v2"
RERANKER_ID  = "nvidia/llama-nemotron-rerank-vl-1b-v2"
RETRIEVE_K   = 50   # FAISS retrieves this many candidates
RERANK_TOP_N = 3    # reranker picks the best N from those 50

PROVIDERS = {
    "gemini": {
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env":       "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "huggingface": {
        "base_url":      "https://api-inference.huggingface.co/v1/",
        "key_env":       "HUGGINGFACE_API_KEY",
        "default_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "groq": {
        "base_url":      "https://api.groq.com/openai/v1/",
        "key_env":       "GROQ_API_KEY",
        "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "openrouter": {
        "base_url":      "https://openrouter.ai/api/v1/",
        "key_env":       "OPENROUTER_API_KEY",
        "default_model": "google/gemma-4-31b-it:free",  # best free vision model Jun 2026
    },
    "nvidia": {
        "base_url":      "https://integrate.api.nvidia.com/v1/",
        "key_env":       "NVIDIA_API_KEY",
        "default_model": "qwen/qwen2.5-vl-7b-instruct",
    },
}


def load_env():
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_client():
    provider_name = os.environ.get("PROVIDER", "openrouter").lower()
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider_name}'. Choose from: {', '.join(PROVIDERS)}")
    cfg = PROVIDERS[provider_name]
    api_key = os.environ.get(cfg["key_env"], "")
    model = os.environ.get("MODEL", cfg["default_model"])
    if not api_key or api_key == "your_key_here":
        raise ValueError(f"API key not set. Set {cfg['key_env']} in .env")
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
    print(f"[reader] {provider_name} — {model}")
    return client, model


def load_index():
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError("Index not found. Run index.py first.")
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "manifest.json") as f:
        manifest = json.load(f)
    return index, manifest


def embed_query(query: str, processor, model) -> np.ndarray:
    device = next(model.parameters()).device
    inputs = processor(
        text=[f"Query: {query}"],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        emb = outputs.hidden_states[-1].mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype("float32")


def rerank(query: str, candidates: list[dict], processor, reranker) -> list[dict]:
    """
    Cross-encoder reranker: scores each [query, image] pair independently.
    Much more accurate than vector similarity for fine-grained document matching.
    See Nemotron paper Table 6: single-vector + reranker ~= late interaction accuracy.
    """
    scores = []
    device = next(reranker.parameters()).device
    for c in candidates:
        img = Image.open(c["path"]).convert("RGB")
        inputs = processor(
            images=[img],
            text=[query],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = reranker(**inputs, output_hidden_states=True)
            # Cross-encoder outputs a relevance logit; take the last (positive class) score
            if hasattr(outputs, "logits"):
                score = outputs.logits[0].item() if outputs.logits.numel() == 1 \
                    else outputs.logits[0][-1].item()
            else:
                # Fallback: mean pool last hidden state as proxy score
                score = outputs.hidden_states[-1].mean().item()
        scores.append(score)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:RERANK_TOP_N]]


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_vlm(client: OpenAI, model: str, query: str, tile_paths: list) -> str:
    """Send the top-N tiles together so the VLM can reason across all of them."""
    content = []
    for tile_path in tile_paths:
        b64 = image_to_base64(tile_path)
        ext = Path(tile_path).suffix.lstrip(".")
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    content.append({"type": "text", "text": (
        "You are a document analyst. Answer the following question "
        "using only the information visible in the document pages above.\n\n"
        f"Question: {query}"
    )})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def main(query: str):
    load_env()
    client, vlm_model = get_client()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[1/4] Loading embedder ({EMBEDDER_ID}) on {device}...")
    processor = AutoProcessor.from_pretrained(EMBEDDER_ID, trust_remote_code=True)
    embedder = AutoModel.from_pretrained(
        EMBEDDER_ID, trust_remote_code=True, dtype=dtype, device_map="auto"
    ).eval()

    print(f"[2/4] Loading reranker ({RERANKER_ID}) on {device}...")
    re_processor = AutoProcessor.from_pretrained(RERANKER_ID, trust_remote_code=True)
    try:
        reranker = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_ID, trust_remote_code=True, dtype=dtype, device_map="auto"
        ).eval()
    except Exception:
        reranker = AutoModel.from_pretrained(
            RERANKER_ID, trust_remote_code=True, dtype=dtype, device_map="auto"
        ).eval()

    print(f"[3/4] Retrieving top-{RETRIEVE_K} via FAISS, reranking to top-{RERANK_TOP_N}...")
    index, manifest = load_index()
    k = min(RETRIEVE_K, index.ntotal)
    query_emb = embed_query(query, processor, embedder)
    scores, indices = index.search(query_emb, k)
    candidates = [
        {"path": manifest[idx], "faiss_score": float(score)}
        for score, idx in zip(scores[0], indices[0]) if idx >= 0
    ]
    top_pages = rerank(query, candidates, re_processor, reranker)

    print(f"\nTop pages after reranking:")
    for i, p in enumerate(top_pages):
        print(f"  [{i+1}] {Path(p['path']).name}  (FAISS score: {p['faiss_score']:.4f})")

    top_tiles = [p["path"] for p in top_pages]
    print(f"\n[4/4] Asking {vlm_model} to read {[Path(t).name for t in top_tiles]}...")
    answer = ask_vlm(client, vlm_model, query, top_tiles)

    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
