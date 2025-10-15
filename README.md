# FAISS Mastery — Beginner → Advanced

A hands-on FAISS (Facebook AI Similarity Search) notebook and demo that takes you from basics to production-ready topics: index types, distance metrics, ANN tuning, GPU acceleration, PQ compression, HNSW graphs, and an end-to-end semantic search demo using `sentence-transformers`.


---

## Contents

- ✅ Overview of FAISS and when to use it  
- ✅ Exact search (IndexFlatL2)  
- ✅ ANN indexes: IVF, IVFPQ (Product Quantization)  
- ✅ Graph-based search: HNSW  
- ✅ Cosine similarity via normalization + inner product  
- ✅ GPU offloading example (how-to)  
- ✅ Semantic search demo using `sentence-transformers`  
- ✅ Save / load indexes, index tuning experiments (nlist, nprobe, m, nbits, efSearch)  
- ✅ Example code snippets and hands-on exercises

---
Key Concepts (cheat-sheet)

Distance metrics: L2 (Euclidean) via IndexFlatL2, Inner Product via IndexFlatIP. Compute cosine similarity by L2-normalizing vectors then using inner product.

Index types:

IndexFlatL2: exact, no compression.

IndexIVFFlat: IVF — cluster-based ANN; tune nlist (clusters) and nprobe (probed clusters).

IndexIVFPQ: IVF + Product Quantization (compresses vectors). Tune m (subvectors) and nbits.

IndexHNSWFlat: HNSW graph for high-speed ANN; tune efConstruction and efSearch.

GPU: Offload CPU index to GPU with faiss.index_cpu_to_gpu(resources, gpu_id, index) to speed up search for large datasets.

Save/load: faiss.write_index(index, path) and faiss.read_index(path).

Tuning tips

Increase nlist to get finer clustering (needs more training time).

Increase nprobe at query time to improve recall at the cost of latency.

More PQ m (subvectors) increases accuracy but also CPU work.

For HNSW, efSearch controls search thoroughness; efConstruction controls build time vs. accuracy.

Normalize vectors for cosine similarity.

Semantic Search Example (high-level)

Convert documents to embeddings with sentence-transformers.

Normalize embeddings for cosine similarity.

Create IndexFlatIP or IndexHNSWFlat, add embeddings.

Encode query → normalize → index.search(query_emb, k) → return top-k docs.
