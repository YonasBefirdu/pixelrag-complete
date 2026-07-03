"""
Download the NVIDIA embedding and reranker models to your local HuggingFace cache.
Run this once before running index.py or query.py.

Models downloaded (~3GB each):
  - nvidia/llama-nemotron-embed-vl-1b-v2  (embedder, single-vector)
  - nvidia/llama-nemotron-rerank-vl-1b-v2 (reranker, cross-encoder)

Usage: python download_models.py
"""

from transformers import AutoProcessor, AutoModel, AutoModelForSequenceClassification

EMBEDDER_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
RERANKER_ID = "nvidia/llama-nemotron-rerank-vl-1b-v2"

print(f"Downloading {EMBEDDER_ID}...")
print("Each model is ~3GB - this will take a while.\n")

AutoProcessor.from_pretrained(EMBEDDER_ID, trust_remote_code=True)
AutoModel.from_pretrained(EMBEDDER_ID, trust_remote_code=True)
print("OK: Embedder downloaded\n")

print(f"Downloading {RERANKER_ID}...")
AutoProcessor.from_pretrained(RERANKER_ID, trust_remote_code=True)
try:
    AutoModelForSequenceClassification.from_pretrained(RERANKER_ID, trust_remote_code=True)
except Exception:
    AutoModel.from_pretrained(RERANKER_ID, trust_remote_code=True)
print("OK: Reranker downloaded\n")

print("Done. Both models cached locally - index.py and query.py will load them instantly.")
