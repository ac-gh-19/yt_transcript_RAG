import sqlite3
import json
from typing import Optional
import hashlib

DB_PATH = "embeddings_cache.sqlite3"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            video_id TEXT NOT NULL,
            chunker TEXT NOT NULL,
            max_tokens INTEGER NOT NULL,
            overlap_tokens INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            PRIMARY KEY (video_id, chunker, max_tokens, overlap_tokens, chunk_id)
        );
        """)

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_embedding(video_id: str, chunker: str, max_tokens: int, overlap_tokens: int, chunk_id: int):
    with get_conn() as conn:
        row = conn.execute("""
            SELECT text_hash, embedding_json
            FROM embeddings
            WHERE video_id=? AND chunker=? AND max_tokens=? AND overlap_tokens=? AND chunk_id=?
        """, (video_id, chunker, max_tokens, overlap_tokens, chunk_id)).fetchone()

    if not row:
        return None  # cache miss

    text_hash, embedding_json = row
    return text_hash, json.loads(embedding_json)

def save_embedding(video_id: str, chunker: str, max_tokens: int, overlap_tokens: int, chunk_id: int, text_hash: str, embedding: list[float]):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO embeddings
            (video_id, chunker, max_tokens, overlap_tokens, chunk_id, text_hash, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (video_id, chunker, max_tokens, overlap_tokens, chunk_id, text_hash, json.dumps(embedding)))