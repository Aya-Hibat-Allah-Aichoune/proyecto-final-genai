"""Configuración del proyecto. Todo es sobreescribible por variable de entorno."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# langchain-google-genai lee GOOGLE_API_KEY; aceptamos también GEMINI_API_KEY
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ.pop("GEMINI_API_KEY")

DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers_teledeteccion")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
# Modelo principal. El plan gratuito limita gemini-3.5-flash a 20 peticiones/día,
# insuficiente para una demo; la variante lite tiene cuota mucho mayor y responde
# bien a preguntas factuales con citas. Cambiar con LLM_MODEL=gemini-3.5-flash si se
# dispone de cuota de pago.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
# Modelo ligero para reescribir la pregunta con el historial: cuota por minuto
# independiente del modelo principal, y la tarea es sencilla.
CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "gemini-3.1-flash-lite")
TOP_K = int(os.getenv("TOP_K", 5))
