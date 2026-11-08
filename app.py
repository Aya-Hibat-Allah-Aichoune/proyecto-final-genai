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

st.set_page_config(page_title="Asistente experto · Teledetección agrícola", page_icon="🌸", layout="centered")

# Tema rosa: los colores base están en .streamlit/config.toml; aquí van los detalles
# (cabecera con degradado, burbujas redondeadas, botones tipo píldora, tarjetas).
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Quicksand', sans-serif; }
    .stApp { background: linear-gradient(180deg, #fff6fa 0%, #ffeaf3 100%); }
    section[data-testid="stSidebar"] { background: #ffe1ee; border-right: 2px solid #f9c6da; }
    section[data-testid="stSidebar"] * { color: #5a2d44; }
    .hero { text-align: center; padding: 1.2rem 1rem 0.6rem; }
    .hero h1 {
        font-size: 2.3rem; font-weight: 700; margin: 0;
        background: linear-gradient(90deg, #ff6fa5, #e75480, #c2185b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #8a5570; margin-top: .4rem; font-size: .95rem; }
    .hero .deco { font-size: 1.6rem; letter-spacing: .4rem; }
    div[data-testid="stChatMessage"] {
        border-radius: 22px; padding: .9rem 1.1rem; margin-bottom: .6rem;
        border: 1.5px solid #f6bfd6; background: #ffffffcc;
        box-shadow: 0 4px 14px rgba(231, 84, 128, .10);
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #ffe1ee; border-color: #f4a6c6;
    }
    div[data-testid="stChatMessageAvatarAssistant"],
    div[data-testid="stChatMessageAvatarUser"] { background: #e75480 !important; color: white !important; }
    div[data-testid="stChatInput"] > div { border: 2px solid #f4a6c6 !important; border-radius: 18px !important; background: #fff; }
    div[data-testid="stChatInput"] textarea { padding-left: 1rem !important; }
    .stButton > button {
        border-radius: 999px; border: 0; padding: .45rem 1.2rem; font-weight: 700;
        background: linear-gradient(90deg, #ff8fb8, #e75480); color: #fff;
        box-shadow: 0 4px 12px rgba(231, 84, 128, .25);
    }
    .stButton > button:hover { background: linear-gradient(90deg, #ff7aab, #d13d6f); color: #fff; }
    details[data-testid="stExpander"] { border: 1.5px solid #f6bfd6; border-radius: 16px; background: #fff8fb; }
    details[data-testid="stExpander"] summary { color: #c2185b; font-weight: 700; }
    .stCaption, small { color: #a06a85 !important; }
    .source-card {
        background: #fff; border-left: 4px solid #e75480; border-radius: 12px;
        padding: .5rem .8rem; margin: .35rem 0; font-size: .9rem;
    }
    .source-card b { color: #c2185b; }
    .kb-doc { background: #fff; border-radius: 14px; padding: .55rem .75rem; margin: .4rem 0; border: 1px solid #f6bfd6; }
    .kb-doc b { color: #c2185b; }
    .kb-doc small { color: #8a5570 !important; }
    </style>
    <div class="hero">
      <div class="deco">🌸 🌾 🌸</div>
      <h1>Asistente experto en teledetección agrícola</h1>
      <p>RAG con Gemini + ChromaDB + LangGraph. Responde solo con los artículos indexados y cita documento y página.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando base de conocimiento y agente...")
def load_agent():
    store = build_knowledge_base()
    return build_agent(store), describe_collection(store)


agent, info = load_agent()

with st.sidebar:
    st.markdown("### 💗 Base de conocimiento")
    st.write(f"{info['n_chunks']} fragmentos de {len(info['documents'])} artículos")
    for src, d in info["documents"].items():
        st.markdown(f"<div class='kb-doc'><b>{src}</b> · {d['pages']} pág.<br><small>{d['title']}</small></div>", unsafe_allow_html=True)
    st.markdown("")
    if st.button("✨ Nueva conversación"):
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
        with st.expander("📚 Fuentes"):
            for s in turn["sources"]:
                st.markdown(f"<div class='source-card'><b>[{s['n']}]</b> {s['source']}, página {s['page']} <small>(sim. {s['score']})</small><br><small>{s['excerpt']}…</small></div>", unsafe_allow_html=True)

if question := st.chat_input("Pregunta sobre los artículos (p. ej. ¿qué índice detecta mejor el estrés hídrico en tomate?)"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("🌸 Buscando en los artículos..."):
            result = ask(agent, question, thread_id=st.session_state.thread_id)
        st.write(result["answer"])
        if result["standalone_question"] != question:
            st.caption(f"Pregunta interpretada con el historial: _{result['standalone_question']}_")
        with st.expander("📚 Fuentes"):
            for s in result["sources"]:
                st.markdown(f"<div class='source-card'><b>[{s['n']}]</b> {s['source']}, página {s['page']} <small>(sim. {s['score']})</small><br><small>{s['excerpt']}…</small></div>", unsafe_allow_html=True)
    st.session_state.turns.append(result)
