"""Chat por terminal con memoria. Escribe 'salir' para terminar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import ask, build_agent  # noqa: E402

if __name__ == "__main__":
    agent = build_agent()
    print("Asistente listo (escribe 'salir' para terminar).\n")
    while True:
        q = input("Tú: ").strip()
        if q.lower() in {"salir", "exit", "quit"}:
            break
        if not q:
            continue
        r = ask(agent, q, thread_id="cli")
        print(f"\nAsistente: {r['answer']}")
        print("Fuentes: " + "; ".join(f"[{s['n']}] {s['source']} p.{s['page']}" for s in r["sources"]) + "\n")
