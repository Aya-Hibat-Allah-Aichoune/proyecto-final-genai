"""Agente experto en LangGraph: RAG sobre ChromaDB + Gemini + memoria de conversación.

Grafo:
    START -> condense -> retrieve -> generate -> END

- condense:  si hay historial, reescribe la última pregunta como una pregunta
             autónoma (resuelve "ese índice", "y en el trigo?") para que la
             recuperación también se beneficie de la memoria.
- retrieve:  top-k chunks de Chroma para la pregunta (reescrita).
- generate:  Gemini responde con el system prompt, el contexto numerado y el
             historial de la conversación. Cita [n] y se abstiene si no hay base.

Memoria: `MemorySaver` de LangGraph guarda el estado por `thread_id`; cada hilo
es una conversación independiente. Se recortan los mensajes antiguos para no
crecer sin límite.
"""
from __future__ import annotations

import re
import time
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import CONDENSE_MODEL, LLM_MODEL, TOP_K
from .knowledge_base import get_vectorstore, retrieve

SYSTEM_PROMPT = """Eres un asistente experto en teledetección hiperespectral aplicada a la agricultura \
(índices de vegetación, estrés hídrico, plagas y enfermedades de cultivos). Tu conocimiento proviene \
EXCLUSIVAMENTE de los fragmentos de artículos científicos que se te proporcionan en cada turno.

Reglas:
1. Responde solo con información contenida en los fragmentos. Si no contienen lo necesario, dilo \
claramente ("Los documentos no contienen información sobre...") y no inventes ni completes con \
conocimiento general.
2. Cita el fragmento que sostiene cada afirmación con el formato [n] (por ejemplo [2]); puedes citar \
varios: [1][3].
3. Responde en el mismo idioma en que está escrita la pregunta.
4. Mantén el hilo de la conversación: si la pregunta se refiere a algo mencionado antes ("ese índice", \
"el mismo cultivo"), resuélvelo con el historial. Si hace referencia a algo que no aparece ni en el \
historial ni en los fragmentos, no supongas a qué se refiere: pide que lo concrete.
5. Tono técnico y preciso, sin relleno: entre dos y seis frases, sin introducciones ni despedidas. \
Usa listas solo cuando ayuden a comparar."""

CONDENSE_PROMPT = """Dada la conversación anterior y la última pregunta del usuario, reescribe la última \
pregunta como una pregunta autónoma y completa, en su mismo idioma, resolviendo referencias como \
"ese índice" o "el mismo cultivo" con la información del historial. Si ya es autónoma, devuélvela tal cual. \
Devuelve solo la pregunta reescrita, sin explicaciones."""

MAX_HISTORY_MESSAGES = 10  # 5 turnos pregunta-respuesta


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str                 # última pregunta tal como la escribió el usuario
    standalone_question: str      # pregunta reescrita con el historial
    context: list[Document]       # chunks recuperados
    sources: list[dict]           # fuentes serializables para mostrar


_RETRY_DELAY = re.compile(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"]?(\d+)", re.I)


def invoke_with_retry(llm, msgs, retries: int = 4):
    """Llama al modelo y, ante 429/503 (cuota por minuto del plan gratuito o
    servicio saturado), espera lo que indica la API y reintenta."""
    for attempt in range(retries):
        try:
            return llm.invoke(msgs)
        except Exception as exc:
            text = str(exc)
            transient = any(t in text for t in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not transient or attempt == retries - 1:
                raise
            m = _RETRY_DELAY.search(text)
            delay = float(next(g for g in m.groups() if g)) + 1 if m else 15 * (attempt + 1)
            time.sleep(min(delay, 90))


def message_text(msg: BaseMessage) -> str:
    """Texto plano de un mensaje: las versiones recientes de langchain-google-genai
    devuelven `content` como lista de bloques en lugar de str."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, página {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def build_agent(store=None, llm: ChatGoogleGenerativeAI | None = None, k: int = TOP_K,
                condense_llm: ChatGoogleGenerativeAI | None = None):
    store = store or get_vectorstore()
    # thinking_budget=0: respuestas cortas y con citas no necesitan razonamiento
    # extendido, y así tardan ~1-3 s en vez de 5-10 s.
    llm = llm or ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, thinking_budget=0)
    condense_llm = condense_llm or (llm if LLM_MODEL == CONDENSE_MODEL or not isinstance(llm, ChatGoogleGenerativeAI)
                                    else ChatGoogleGenerativeAI(model=CONDENSE_MODEL, temperature=0.0, thinking_budget=0))

    def condense(state: AgentState) -> dict:
        question = state["messages"][-1].content
        history = [m for m in state["messages"][:-1] if isinstance(m, (HumanMessage, AIMessage))]
        if not history:
            return {"question": question, "standalone_question": question}
        history = trim_messages(history, max_tokens=MAX_HISTORY_MESSAGES, token_counter=len,
                                strategy="last", start_on="human")
        msgs = [SystemMessage(CONDENSE_PROMPT), *history, HumanMessage(f"Última pregunta: {question}")]
        rewritten = message_text(invoke_with_retry(condense_llm, msgs)).strip() or question
        return {"question": question, "standalone_question": rewritten}

    def retrieve_node(state: AgentState) -> dict:
        hits = retrieve(store, state["standalone_question"], k=k)
        docs = [d for d, _ in hits]
        sources = [
            {"n": i, "source": d.metadata["source"], "title": d.metadata["title"],
             "page": d.metadata["page"], "score": round(float(s), 3), "excerpt": d.page_content[:200]}
            for i, (d, s) in enumerate(hits, start=1)
        ]
        return {"context": docs, "sources": sources}

    def generate(state: AgentState) -> dict:
        history = [m for m in state["messages"][:-1] if isinstance(m, (HumanMessage, AIMessage))]
        history = trim_messages(history, max_tokens=MAX_HISTORY_MESSAGES, token_counter=len,
                                strategy="last", start_on="human")
        user_turn = (
            f"Fragmentos recuperados:\n\n{format_context(state['context'])}\n\n"
            f"Pregunta: {state['question']}\n\n"
            "(Responde en el mismo idioma en que está escrita la pregunta.)"
        )
        msgs = [SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(user_turn)]
        answer = message_text(invoke_with_retry(llm, msgs)).strip()
        return {"messages": [AIMessage(content=answer)]}

    graph = StateGraph(AgentState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=MemorySaver())


def ask(agent, question: str, thread_id: str = "default") -> dict:
    """Un turno de conversación en el hilo `thread_id`. Devuelve respuesta, fuentes
    y la pregunta reescrita (útil para ver cómo actúa la memoria)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"messages": [HumanMessage(question)]}, config=config)
    return {
        "question": question,
        "standalone_question": state["standalone_question"],
        "answer": message_text(state["messages"][-1]),
        "sources": state["sources"],
    }


def history(agent, thread_id: str = "default") -> list[BaseMessage]:
    snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values.get("messages", []) if snapshot and snapshot.values else []
