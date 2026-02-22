from ingest import fetch_transcript, get_video_id
from chunker import chunk_by_token_budget
from embeddings import embed_texts
from retrieval import cosine_top_k
from cache_db import init_db, hash_text, load_embedding, save_embedding

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

if __name__ == "__main__":
    init_db()

    url = "https://www.youtube.com/watch?v=UCP_be8pT50"
    query = "What should I do when I face procrastination?"

    # chunking config
    chunker_name = "token_budget"
    max_tokens = 200
    overlap_tokens = 15

    video_id = get_video_id(url)

    segments = fetch_transcript(url)
    # (chunk_id, text, start, end, segment_range, token_esetimate)
    chunks = chunk_by_token_budget(segments, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    # Build doc embeddings with cache
    chunk_vecs = [None] * len(chunks)
    to_embed_texts = []
    to_embed_slots = []

    for i, c in enumerate(chunks):
        text = c["text"]
        hashed_text = hash_text(text)

        cached = load_embedding(video_id, chunker_name, max_tokens, overlap_tokens, c["chunk_id"])
        if cached and cached[0] == hashed_text:
            # cached returns (text_hash, embedding)
            chunk_vecs[i] = cached[1]
        else:
            to_embed_texts.append(text)
            to_embed_slots.append((i, c["chunk_id"], hashed_text))

    # Batch-embed only cache misses
    if to_embed_texts:
        new_vecs = embed_texts(to_embed_texts)
        print(f"Embedded {len(new_vecs)} new chunks (cache misses), saving to cache...")
        for (i, chunk_id, hashed_text), vec in zip(to_embed_slots, new_vecs):
            save_embedding(video_id, chunker_name, max_tokens, overlap_tokens, chunk_id, hashed_text, vec)
            chunk_vecs[i] = vec

    # Embed query to retrieve similar
    query_vec = embed_texts([query])[0]

    top = cosine_top_k(query_vec, chunk_vecs, k=5)

    for rank, (idx, score) in enumerate(top, 1):
        c = chunks[idx]
        print(f"{rank}. score={score:.4f}  [{fmt_time(c['start'])}-{fmt_time(c['end'])}]  chunk_id={c['chunk_id']}")
        print("   ", c["text"].replace("\n", " "))
        print()