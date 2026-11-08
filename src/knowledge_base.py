"""Base de conocimiento vectorial: ChromaDB persistente + Gemini Embeddings.

- Una colección por dominio (`papers_teledeteccion`), persistida en `chroma_db/`
  para no volver a pagar los embeddings en cada arranque.
- Los chunks se añaden en lotes pequeños con reintentos: el plan gratuito de
  Gemini limita las peticiones de embeddings por minuto.
- Cada chunk guarda `source`, `title`, `page` y `chunk_id` para citar.
"""
from __future__ import annotations

import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from .ingest import load_documents


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # task_type se deja sin fijar: la librería usa RETRIEVAL_DOCUMENT para los
    # pasajes y RETRIEVAL_QUERY para las preguntas, que es lo recomendado.
    kwargs = {"model": EMBEDDING_MODEL}
    if "output_dimensionality" in GoogleGenerativeAIEmbeddings.model_fields:
        kwargs["output_dimensionality"] = EMBEDDING_DIM
    return GoogleGenerativeAIEmbeddings(**kwargs)


def get_vectorstore(persist_dir: Path = CHROMA_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _add_with_retry(store: Chroma, batch: list[Document], ids: list[str], retries: int = 6) -> None:
    for attempt in range(retries):
        try:
            store.add_documents(batch, ids=ids)
            return
        except Exception as exc:  # 429 / 503 del plan gratuito
            msg = str(exc)
            if attempt == retries - 1 or not any(code in msg for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                raise
            time.sleep(min(10 * 2 ** attempt, 90))


def build_knowledge_base(docs_dir: Path = DOCS_DIR, persist_dir: Path = CHROMA_DIR,
                         batch_size: int = 25, rebuild: bool = False) -> Chroma:
    """Indexa los PDF de `docs_dir` en Chroma. Si la colección ya tiene datos y
    `rebuild` es False, la reutiliza tal cual."""
    store = get_vectorstore(persist_dir)
    existing = store._collection.count()
    if existing and not rebuild:
        print(f"Colección '{COLLECTION_NAME}' ya existe con {existing} chunks; se reutiliza.")
        return store
    if existing:
        store.delete_collection()
        store = get_vectorstore(persist_dir)

    docs = load_documents(docs_dir)
    print(f"{len(docs)} chunks de {len({d.metadata['source'] for d in docs})} documentos. Calculando embeddings...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        _add_with_retry(store, batch, ids=[d.metadata["chunk_id"] for d in batch])
        print(f"  {min(i + batch_size, len(docs))}/{len(docs)}", end="\r")
    print(f"\nIndexados {store._collection.count()} chunks en {persist_dir}")
    return store


def retrieve(store: Chroma, query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Top-k chunks con su similitud coseno (1 = idéntico)."""
    return [(d, 1 - dist) for d, dist in store.similarity_search_with_score(query, k=k)]


def describe_collection(store: Chroma) -> dict:
    data = store._collection.get(include=["metadatas"])
    per_doc: dict[str, dict] = {}
    for m in data["metadatas"]:
        d = per_doc.setdefault(m["source"], {"title": m["title"], "chunks": 0, "pages": 0})
        d["chunks"] += 1
        d["pages"] = max(d["pages"], m["page"])
    return {"collection": COLLECTION_NAME, "n_chunks": len(data["ids"]), "documents": per_doc}
