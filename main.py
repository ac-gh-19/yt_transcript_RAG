from ingest import fetch_transcript
from chunker import chunk_by_token_budget  # your function
from embeddings import embed_texts
from retrieval import cosine_top_k

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=UCP_be8pT50"
    query = "Why causes procrastination?"

    segments = fetch_transcript(url)
    chunks = chunk_by_token_budget(segments, max_tokens=200, overlap_tokens=15)

    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = embed_texts(chunk_texts)
    query_vec = embed_texts([query])[0]

    top = cosine_top_k(query_vec, chunk_vecs, k=5)

    for rank, (idx, score) in enumerate(top, 1):
        c = chunks[idx]
        print(f"{rank}. score={score:.4f}  [{fmt_time(c['start'])}-{fmt_time(c['end'])}]  chunk_id={c['chunk_id']}")
        print("   ", c["text"].replace("\n", " "))
        print()