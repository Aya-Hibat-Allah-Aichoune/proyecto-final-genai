# Asistente experto en teledetección hiperespectral agrícola

Proyecto final del módulo de IA Generativa: un agente que responde preguntas sobre un dominio técnico usando su propia base de conocimiento vectorial, con **Gemini** como LLM y modelo de embeddings, **ChromaDB** como base vectorial, **LangGraph** como framework de agente y **memoria de conversación**. Se usa desde un notebook y, opcionalmente, desde una interfaz Streamlit.

## 1. Dominio elegido

Documentación técnica: cinco artículos científicos de acceso abierto (MDPI *Sensors* y *Frontiers in Plant Science*, 82 páginas en total) sobre el uso de índices de vegetación hiperespectrales en agricultura:

| Documento | Tema |
|---|---|
| `sensors-19-00375.pdf` | Índice de razón RVI746/670 para detectar el daño del saltahojas marrón en arroz |
| `sensors-21-05705.pdf` | Estimación del estrés hídrico en tomate de invernadero con reflectancia hiperespectral |
| `sensors-24-01916.pdf` | Índices hiperespectrales para evaluar la gravedad del mildiu en *Brassica* |
| `sensors-25-03817.pdf` | NDVI y otros índices como características para reconocer y segmentar cultivos |
| `fpls-08-01219.pdf` | Índices de cubierta para evaluar el estado hídrico del trigo con oídio |

Elegí este dominio porque es texto técnico real, lleno de siglas, cifras y nombres de índices, donde se nota si el RAG recupera y cita con precisión o si el modelo rellena con conocimiento general. Además los documentos están en inglés y las preguntas pueden hacerse en español, lo que pone a prueba la recuperación multilingüe.

## 2. Instalación y ejecución

Requisitos: Python 3.11 o superior y una clave de la API de Gemini (gratuita en [Google AI Studio](https://aistudio.google.com/apikey)).

```bash
python -m venv .venv && source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # y escribir GOOGLE_API_KEY=... dentro
```

La clave se lee siempre de `.env` o del entorno, nunca va en el código. `.env` está en `.gitignore`.

**Notebook (entregable principal):**

```bash
jupyter notebook notebooks/proyecto_final.ipynb
```

El notebook construye la base de conocimiento la primera vez (unos 3 minutos: 352 chunks enviados en lotes de 25, porque el plan gratuito limita las peticiones de embeddings por minuto) y la reutiliza en ejecuciones posteriores desde `chroma_db/`. Se puede construir también desde terminal con `python scripts/build_kb.py` (añadir `--rebuild` para regenerarla).

**Interfaz Streamlit (bonus):**

```bash
streamlit run app.py
```

**Otros:** `python scripts/chat.py` (chat por terminal con memoria) y `pytest` (tests sin red: chunking, limpieza y grafo con LLM simulado).

## 3. Cómo funciona

```
PDF ──pypdf──▶ páginas ──chunking──▶ chunks + metadatos ──Gemini Embeddings──▶ ChromaDB
                                                                                  │
pregunta ──▶ [condense] ──▶ [retrieve] ──▶ [generate] ──▶ respuesta con citas [n] + fuentes
                 ▲ historial (MemorySaver, por thread_id)        ▲ historial
```

- **Ingesta** (`src/ingest.py`): extracción por página, limpieza de guiones y saltos de línea, chunks de ~1.200 caracteres con solapamiento de 200, corte preferente en fin de frase y sin cruzar páginas. Cada chunk guarda documento, título y página para poder citar.
- **Base de conocimiento** (`src/knowledge_base.py`): colección persistente en ChromaDB con distancia coseno y embeddings `gemini-embedding-001` (768 dimensiones). La librería usa el tipo de tarea `RETRIEVAL_DOCUMENT` para los pasajes y `RETRIEVAL_QUERY` para las preguntas, que es lo recomendado para búsqueda.
- **Agente** (`src/agent.py`): grafo de LangGraph con tres nodos.
  - `condense`: si hay historial, Gemini (`gemini-3.1-flash-lite`) reescribe la última pregunta como una pregunta autónoma ("¿y en qué bandas se basa ese índice?" pasa a "¿en qué bandas se basa el índice RVI746/670?"). Así la memoria mejora también la recuperación.
  - `retrieve`: top-5 chunks de ChromaDB para la pregunta reescrita.
  - `generate`: `gemini-3.1-flash-lite` con temperatura 0,1 y razonamiento extendido desactivado (`thinking_budget=0`), que recibe el system prompt, los fragmentos numerados y el historial.

**Elección del modelo.** Empecé con `gemini-3.5-flash`, pero el plan gratuito lo limita a 20 peticiones al día por proyecto (y 5 por minuto), insuficiente para desarrollar y para una demo en vivo: cada turno del agente hace dos llamadas. La variante lite tiene una cuota diaria mucho mayor y, en las pruebas, responde igual de bien a preguntas factuales con citas. El modelo se cambia con la variable `LLM_MODEL` (por ejemplo `LLM_MODEL=gemini-3.5-flash` con cuota de pago). Los límites exactos de cada cuenta se ven en https://aistudio.google.com/rate-limit; el código reintenta respetando el tiempo de espera que devuelve la API cuando se supera la cuota por minuto.
- **Memoria**: `MemorySaver` de LangGraph guarda el estado de cada conversación por `thread_id`. El historial se recorta a los últimos cinco turnos para acotar el contexto. Es memoria en proceso: para producción se cambiaría por un checkpointer persistente (SQLite o Postgres).

## 4. System prompt y justificación

Prompt completo (`SYSTEM_PROMPT` en `src/agent.py`):

> Eres un asistente experto en teledetección hiperespectral aplicada a la agricultura (índices de vegetación, estrés hídrico, plagas y enfermedades de cultivos). Tu conocimiento proviene EXCLUSIVAMENTE de los fragmentos de artículos científicos que se te proporcionan en cada turno.
>
> Reglas:
> 1. Responde solo con información contenida en los fragmentos. Si no contienen lo necesario, dilo claramente ("Los documentos no contienen información sobre...") y no inventes ni completes con conocimiento general.
> 2. Cita el fragmento que sostiene cada afirmación con el formato [n] (por ejemplo [2]); puedes citar varios: [1][3].
> 3. Responde en el mismo idioma en que está escrita la pregunta.
> 4. Mantén el hilo de la conversación: si la pregunta se refiere a algo mencionado antes ("ese índice", "el mismo cultivo"), resuélvelo con el historial. Si hace referencia a algo que no aparece ni en el historial ni en los fragmentos, no supongas a qué se refiere: pide que lo concrete.
> 5. Tono técnico y preciso, sin relleno: entre dos y seis frases, sin introducciones ni despedidas. Usa listas solo cuando ayuden a comparar.

Decisiones de diseño:

- **Rol y dominio acotados.** Un rol concreto fija el vocabulario técnico y reduce las respuestas genéricas. Decir de dónde viene su conocimiento ("exclusivamente de los fragmentos") prepara la regla principal.
- **Solo fragmentos, y abstención explícita.** Es la regla que evita alucinaciones. Pedir una frase concreta para la abstención hace el comportamiento comprobable: en el notebook, una pregunta sobre LoRaWAN obtiene "Los documentos no contienen información sobre LoRaWAN".
- **Citas `[n]` por afirmación.** Cada número corresponde a un fragmento con documento y página que se muestran junto a la respuesta, así cualquier afirmación se puede verificar. Es también lo que permitiría medir la fidelidad de las respuestas de forma automática.
- **Idioma de la pregunta.** Los artículos están en inglés y las preguntas suelen llegar en español. En pruebas, sin esta regla el modelo contestaba a veces en español a preguntas en inglés.
- **Referencias al historial, y pedir aclaración cuando no hay antecedente.** Hace explícito que la conversación tiene memoria. Se complementa con el nodo `condense`, que reescribe la pregunta antes de recuperar: con el prompt solo, el modelo entendía la referencia pero la recuperación traía fragmentos poco relevantes. La segunda parte de la regla salió de una prueba: en un hilo nuevo, ante "¿en qué bandas se basa ese índice?", el modelo se inventaba un antecedente; ahora pide que se concrete el índice.
- **Formato breve y temperatura baja.** Respuestas de dos a seis frases, temperatura 0,1 y razonamiento extendido desactivado: para preguntas factuales con citas no aportaban calidad y multiplicaban el tiempo de respuesta (de 1 a 3 segundos frente a 5 a 10).
- **Recordatorio del idioma en cada turno.** Además de la regla 3, el mensaje de usuario termina con "(Responde en el mismo idioma en que está escrita la pregunta.)": con el system prompt solo, el modelo lite contestaba en español a preguntas en inglés porque todo el resto del prompt está en español.

Iteraciones: el prompt empezó con las reglas 1 y 2. La regla 3 y el recordatorio por turno se añadieron tras ver respuestas en el idioma equivocado; la 4 y el nodo `condense` tras comprobar que las preguntas de seguimiento recuperaban mal; la 5 para evitar introducciones del tipo "Según los documentos proporcionados...".

## 5. Memoria de conversación

En el notebook, sección 4: tras preguntar por el índice más sensible al saltahojas marrón en arroz, la pregunta "¿Y en qué bandas se basa ese índice?" se reescribe a una pregunta sobre el RVI746/670 y se responde con las bandas de 670 nm y 746 nm; una tercera pregunta ("¿cuántas plantas por maceta usaron en ese experimento?") sigue resolviendo la referencia. La misma pregunta ambigua en un hilo nuevo no puede resolverse y el agente pide que se concrete el índice, lo que demuestra que la memoria es por conversación.

## 6. Preguntas de ejemplo

Seis preguntas documentadas con sus respuestas y fuentes en la sección 5 del notebook: una por documento, una en inglés, una sobre un dato numérico concreto y una fuera del corpus para comprobar la abstención. La sección 5.1 muestra además un caso límite documentado: una pregunta comparativa entre dos artículos.

## 7. Estructura del repositorio

```
notebooks/proyecto_final.ipynb   notebook ejecutado: base de conocimiento, agente, memoria, ejemplos y chat
src/config.py                    configuración (modelos, rutas, chunking), lee .env
src/ingest.py                    PDF -> chunks con metadatos
src/knowledge_base.py            ChromaDB + Gemini Embeddings, construcción y consulta
src/agent.py                     grafo LangGraph, system prompt, memoria
app.py                           interfaz Streamlit (bonus)
scripts/build_kb.py, chat.py     construir la base / chat por terminal
tests/test_project.py            tests sin red
docs/                            los cinco PDF
requirements.txt, .env.example
```

## 8. Limitaciones y siguientes pasos

- Memoria en proceso; se pierde al reiniciar. Siguiente paso: checkpointer persistente.
- Recuperación solo densa y con una única consulta por turno. Una pregunta que exige combinar dos documentos ("compara el tomate y el trigo") recupera fragmentos de uno solo, y el agente responde que no tiene información del otro; es honesto respecto a los fragmentos, pero incompleto respecto al corpus. Siguiente paso: descomponer la pregunta en sub-consultas o reservar un cupo de fragmentos por documento. Con un corpus mayor añadiría también BM25 (siglas y cifras exactas) y un reranker.
- Evaluación cualitativa. Siguiente paso: conjunto de preguntas con respuesta esperada, métricas de recuperación (hit@k, MRR) y de fidelidad de las citas.
- El plan gratuito de Gemini limita las peticiones por minuto y por día; la construcción de la base y el agente reintentan con espera, pero conviene reutilizar `chroma_db/` y no encadenar demasiadas preguntas seguidas.

## 9. Uso de IA

He usado un asistente de IA (Claude Code) durante el desarrollo: para escribir la mayor parte del código a partir de la arquitectura y las decisiones que definí, para depurar la integración con las versiones actuales de LangChain y LangGraph, y para generar el notebook y revisar la redacción. El diseño del dominio, el system prompt y sus iteraciones, la estructura del grafo, los parámetros de chunking y la interpretación de las pruebas son míos, y he revisado y ejecutado todo el código.
