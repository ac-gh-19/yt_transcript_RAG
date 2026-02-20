from ingest import fetch_transcript
from chunker import chunk_by_token_budget

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=UCP_be8pT50"
    snippets = fetch_transcript(url)

    chunks = chunk_by_token_budget(snippets, max_tokens=150, overlap_tokens=30)
    for c in chunks:
        print(f"Chunk {c['chunk_id']} (tokens: {c['token_estimate']}): [{c['start']:.2f}s - {c['end']:.2f}s]")
        print(c['text'])
        print("-" * 40)