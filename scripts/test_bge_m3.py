"""Test BGE-M3 with all 3 embedding types"""
from FlagEmbedding import BGEM3FlagModel

print("Loading BGE-M3...")
m = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')

print("Encoding test...")
o = m.encode(['测试文本', '这是第二个测试句子'], return_dense=True, return_sparse=True, return_colbert_vecs=True)

dense = o['dense_vecs']
sparse = o['lexical_weights']
colbert = o['colbert_vecs']

print(f"Dense vectors: {dense.shape[1]} dimensions")
print(f"Sparse: first text has {len(sparse[0])} lexical tokens, second has {len(sparse[1])}")
print(f"ColBERT: first text has {colbert[0].shape[0]} token embeddings, shape={colbert[0].shape}")
print("\nAll 3 embedding types working!")
