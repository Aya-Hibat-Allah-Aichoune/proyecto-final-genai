"""Interfaz Streamlit (bonus): chat con el agente experto.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import os
import uuid

import streamlit as st

# En Streamlit Community Cloud los secretos llegan por st.secrets; se copian al
# entorno para que src/config.py los lea igual que en local (.env).
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from src.agent import ask, build_agent
from src.knowledge_base import build_knowledge_base, describe_collection

st.set_page_config(page_title="Asistente experto · Teledetección agrícola", page_icon="🌾", layout="centered")
st.title("🌾 Asistente experto en teledetección agrícola")
st.caption("RAG con Gemini + ChromaDB + LangGraph. Responde solo con los artículos indexados y cita documento y página.")


@st.cache_resource(show_spinner="Cargando base de conocimiento y agente...")
def load_agent():
    store = build_knowledge_base()
    return build_agent(store), describe_collection(store)


agent, info = load_agent()

with st.sidebar:
    st.subheader("Base de conocimiento")
    st.write(f"{info['n_chunks']} fragmentos de {len(info['documents'])} artículos")
    for src, d in info["documents"].items():
        st.markdown(f"- **{src}** ({d['pages']} pág.)  \n  <small>{d['title']}</small>", unsafe_allow_html=True)
    if st.button("Nueva conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.turns = []
        st.rerun()

st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("turns", [])

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Fuentes"):
            for s in turn["sources"]:
                st.markdown(f"**[{s['n']}]** {s['source']}, página {s['page']} (sim. {s['score']})  \n<small>{s['excerpt']}…</small>", unsafe_allow_html=True)

if question := st.chat_input("Pregunta sobre los artículos (p. ej. ¿qué índice detecta mejor el estrés hídrico en tomate?)"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los artículos..."):
            result = ask(agent, question, thread_id=st.session_state.thread_id)
        st.write(result["answer"])
        if result["standalone_question"] != question:
            st.caption(f"Pregunta interpretada con el historial: _{result['standalone_question']}_")
        with st.expander("Fuentes"):
            for s in result["sources"]:
                st.markdown(f"**[{s['n']}]** {s['source']}, página {s['page']} (sim. {s['score']})  \n<small>{s['excerpt']}…</small>", unsafe_allow_html=True)
    st.session_state.turns.append(result)
