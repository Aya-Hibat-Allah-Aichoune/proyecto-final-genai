"""Construye (o reconstruye con --rebuild) la base de conocimiento en chroma_db/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_base import build_knowledge_base, describe_collection  # noqa: E402

if __name__ == "__main__":
    store = build_knowledge_base(rebuild="--rebuild" in sys.argv)
    info = describe_collection(store)
    print(f"\nColección '{info['collection']}': {info['n_chunks']} chunks")
    for src, d in info["documents"].items():
        print(f"  {src:28s} {d['pages']:3d} pág. {d['chunks']:4d} chunks  {d['title'][:60]}")
