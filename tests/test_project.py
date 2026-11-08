"""Tests sin red: chunking, limpieza y estructura del grafo (con LLM y vector store simulados)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from src.ingest import clean_text, split_text  # noqa: E402


def test_split_text_size_and_overlap():
    text = ("Una frase de prueba con algo de contenido. " * 200).strip()
    chunks = split_text(text, size=500, overlap=100)
    assert len(chunks) > 1 and all(len(c) <= 500 for c in chunks)
    assert chunks[1][:40] in chunks[0]


def test_clean_text():
    assert clean_text("measure-\nment of soil\nmoisture") == "measurement of soil moisture"


class FakeStore:
    def similarity_search_with_score(self, query, k=5):
        return [(Document(page_content=f"Texto sobre {query}", metadata={"source": "a.pdf", "title": "A", "page": 1,
                                                                            "chunk_id": "a:p1:c0"}), 0.1)]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, msgs):
        self.calls.append(msgs)
        text = msgs[-1].content
        if "Última pregunta" in text:
            return AIMessage(content="pregunta reescrita")
        return AIMessage(content="Respuesta [1].")


def test_graph_runs_with_memory():
    from src.agent import ask, build_agent, history

    llm = FakeLLM()
    agent = build_agent(store=FakeStore(), llm=llm)
    r1 = ask(agent, "¿Qué es X?", thread_id="t1")
    assert r1["answer"] == "Respuesta [1]." and r1["standalone_question"] == "¿Qué es X?"
    assert r1["sources"][0]["page"] == 1
    r2 = ask(agent, "¿Y eso?", thread_id="t1")
    assert r2["standalone_question"] == "pregunta reescrita"      # el historial activó la reescritura
    assert len(history(agent, "t1")) == 4                          # 2 preguntas + 2 respuestas
    assert len(history(agent, "otro")) == 0                        # hilos independientes
