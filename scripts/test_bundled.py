"""Test bundled Python with local model"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

print("Loading BGE-M3 from local...")
m3 = BGEM3FlagModel(
    os.path.join(model_path, "BAAI--bge-m3"),
    use_fp16=False, device="cpu"
)
o = m3.encode(["test"], return_dense=True)
print(f"  BGE-M3: {o['dense_vecs'].shape[1]}d OK")

print("Loading Reranker from local...")
rerank = FlagReranker(
    os.path.join(model_path, "BAAI--bge-reranker-v2-m3"),
    use_fp16=False, device="cpu"
)
scores = rerank.compute_score([["test query", "test doc"]], normalize=True)
print(f"  Reranker: {scores} OK")

print("\nBundled Python fully working!")
