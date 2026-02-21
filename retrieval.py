import numpy as np

def cosine_top_k(query_vec: list[float], doc_vecs: list[list[float]], k: int = 5):
    Q = np.array(query_vec, dtype=np.float32)
    M = np.array(doc_vecs, dtype=np.float32)

    Q = Q / (np.linalg.norm(Q) + 1e-8)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    sims = M @ Q
    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top_idx]