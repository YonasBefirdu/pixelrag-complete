"""
Step 2 — Embed tiles and build a FAISS index.
Usage: python index.py
Reads tiles from ./tiles/, writes index to ./index/

Model: nvidia/llama-nemotron-embed-vl-1b-v2
  - 1.4B params, ~3GB RAM — feasible on CPU
  - Single-vector per page (not late interaction) — fast FAISS search
  - Fine-tuned for visual document retrieval (ViDoRe benchmark)
  - Why not Qwen3-VL-Embedding: not fine-tuned for enterprise docs, only web/Wikipedia

Why single-vector (not late interaction):
  Late interaction stores ~773 vectors per page. For 50K pages that is ~185GB.
  Single vector stores 1 vector per page. For 50K pages that is ~200MB.
  Accuracy gap is recovered by the reranker in query.py (see Nemotron paper Table 6).
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import faiss
from transformers import AutoProcessor, AutoModel

TILES_DIR = Path("tiles")
INDEX_DIR = Path("index")
CHECKPOINT_FILE = INDEX_DIR / "checkpoint.npz"
EMBEDDER_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
BATCH_SIZE = 4   # keep low for CPU
MAX_TILES = None  # set to e.g. 30 for quick testing, None to index all
CHECKPOINT_EVERY = 10  # save progress every N tiles


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBEDDER_ID} on {device}...")
    processor = AutoProcessor.from_pretrained(EMBEDDER_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBEDDER_ID,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    ).eval()
    return processor, model, device


def embed_tiles(processor, model, device=None):
    tile_paths = sorted(
        list(TILES_DIR.rglob("*.png")) + list(TILES_DIR.rglob("*.jpg"))
    )
    if not tile_paths:
        raise FileNotFoundError(
            f"No tiles found in {TILES_DIR}/. Run render.py first."
        )

    if MAX_TILES is not None:
        tile_paths = tile_paths[:MAX_TILES]
        print(f"[TEST MODE] Capped at {MAX_TILES} tiles.")

    # Load checkpoint if it exists
    INDEX_DIR.mkdir(exist_ok=True)
    done_paths = []
    all_embeddings = []
    if CHECKPOINT_FILE.exists():
        ckpt = np.load(CHECKPOINT_FILE, allow_pickle=True)
        all_embeddings = list(ckpt["embeddings"])
        done_paths = list(ckpt["paths"])
        print(f"[RESUME] Loaded checkpoint: {len(done_paths)}/{len(tile_paths)} tiles already done.")
        tile_paths = [p for p in tile_paths if str(p) not in done_paths]

    device = next(model.parameters()).device
    print(f"[2/3] Embedding {len(tile_paths)} tiles on {device}...")

    for i, tile_path in enumerate(tile_paths):
        img = Image.open(tile_path).convert("RGB")

        inputs = processor(
            images=[img],
            text=["Represent this document page for retrieval:"],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            emb = outputs.hidden_states[-1].mean(dim=1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu().numpy())
            done_paths.append(str(tile_path))

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(tile_paths):
            np.savez(CHECKPOINT_FILE, embeddings=all_embeddings, paths=done_paths)
            print(f"  {len(done_paths)} tiles embedded (checkpoint saved)")

    all_paths = [Path(p) for p in done_paths]
    return np.vstack(all_embeddings).astype("float32"), all_paths


def build_index(embeddings, tile_paths):
    INDEX_DIR.mkdir(exist_ok=True)
    dim = embeddings.shape[1]

    print(f"[3/3] Building FAISS index (dim={dim}, {len(tile_paths)} vectors)...")
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vectors
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    manifest = [str(p) for p in tile_paths]
    with open(INDEX_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = embeddings.nbytes / 1_000_000
    print(f"[DONE] Index saved to ./{INDEX_DIR}/")
    print(f"       {index.ntotal} vectors, dim={dim}, size={size_mb:.1f} MB")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("      Checkpoint cleaned up.")


if __name__ == "__main__":
    processor, model, device = load_model()
    embeddings, tile_paths = embed_tiles(processor, model)
    build_index(embeddings, tile_paths)
