"""Test that local model loading works offline"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from FlagEmbedding import BGEM3FlagModel, FlagReranker

print("Testing BGE-M3 from local path...")
m = BGEM3FlagModel("models/BAAI--bge-m3", use_fp16=False, device="cpu")
o = m.encode(["测试文本 local model test"], return_dense=True, return_sparse=True, return_colbert_vecs=True)
print(f"  Dense: {o['dense_vecs'].shape[1]}d")
print(f"  Sparse: {len(o['lexical_weights'][0])} lexical tokens")
print(f"  ColBERT: {o['colbert_vecs'][0].shape[0]} tokens x {o['colbert_vecs'][0].shape[1]}d")
print("  BGE-M3: OK")

print("\nTesting BGE-Reranker-v2-m3 from local path...")
r = FlagReranker("models/BAAI--bge-reranker-v2-m3", use_fp16=False, device="cpu")
scores = r.compute_score([["test query", "test document about manufacturing"]], normalize=True)
print(f"  Score: {scores}")
print("  Reranker: OK")

print("\nAll models load from local paths!")
