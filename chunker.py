from typing import List, Dict, Any, Tuple

def estimate_tokens(text: str) -> int:
    # Simple approximation: tokens ≈ 1.3 * words
    words = len(text.split())
    return int(words * 1.3)

def chunk_by_token_budget(
    segments: List[Dict[str, Any]],
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        overlap_tokens = max_tokens - 1

    chunks: List[Dict[str, Any]] = []
    n = len(segments)
    i = 0
    chunk_id = 0

    while i < n:
        start_i = i
        token_sum = 0
        j = i  # j will be the first index NOT included

        # Build chunk [start_i, j)
        while j < n:
            text_j = (segments[j].get("text") or "").strip()
            seg_tokens = estimate_tokens(text_j)

            # If a single segment is huge, force it into its own chunk
            if j == start_i and seg_tokens > max_tokens:
                j += 1
                token_sum = seg_tokens
                break

            if token_sum + seg_tokens <= max_tokens:
                token_sum += seg_tokens
                j += 1
            else:
                break

        if j == start_i:
            # Ensure progress
            j = min(start_i + 1, n)

        chunk_segs = segments[start_i:j]
        chunk_text = " ".join((s.get("text") or "").strip() for s in chunk_segs).strip()
        chunk_start = float(min(s["start"] for s in chunk_segs))
        chunk_end = float(max(s["end"] for s in chunk_segs))

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start": chunk_start,
            "end": chunk_end,
            "segment_range": (start_i, j - 1),
            "token_estimate": token_sum,
        })
        chunk_id += 1

        # Compute overlap tail: keep last segments whose token sum <= overlap_tokens
        if overlap_tokens == 0:
            i = j
            continue

        tail_tokens = 0
        tail_start = j
        k = j - 1
        while k >= start_i:
            text_k = (segments[k].get("text") or "").strip()
            t = estimate_tokens(text_k)
            if tail_tokens + t > overlap_tokens:
                break
            tail_tokens += t
            tail_start = k
            k -= 1

        i = tail_start if tail_start < j else j
        if i <= start_i:
            # If overlap is too aggressive and we'd get stuck, force forward progress.
            i = j

    return chunks

def chunk_by_token_budget_no_overlap(
    segments: List[Dict[str, Any]],
    max_tokens: int = 500,
) -> List[Dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    chunks = []
    n = len(segments)
    i = 0
    chunk_id = 0

    while i < n:
        start_i = i
        token_sum = 0
        j = i

        while j < n:
            text_j = (segments[j].get("text") or "").strip()
            seg_tokens = estimate_tokens(text_j)

            # force single huge segment into its own chunk
            if j == start_i and seg_tokens > max_tokens:
                token_sum = seg_tokens
                j += 1
                break

            if token_sum + seg_tokens <= max_tokens:
                token_sum += seg_tokens
                j += 1
            else:
                break

        if j == start_i:
            j = min(start_i + 1, n)

        chunk_segs = segments[start_i:j]
        chunk_text = " ".join((s.get("text") or "").strip() for s in chunk_segs).strip()

        chunk_start = float(min(s["start"] for s in chunk_segs))
        chunk_end = float(max(s["end"] for s in chunk_segs))

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start": chunk_start,
            "end": chunk_end,
            "segment_range": (start_i, j - 1),
            "token_estimate": token_sum,
        })
        chunk_id += 1
        i = j  # ✅ no overlap

    return chunks