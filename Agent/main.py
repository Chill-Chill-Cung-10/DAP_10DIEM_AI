"""CLI entrypoint (quiet logging)."""

from typing import List

from Agent import app, logger, OPENAI_MODEL, PG_DB, PG_HOST, PG_PORT, test_pg_connection, append_chat_history


def run_cli() -> None:
    print("=" * 60)
    print("  LangGraph Agent  –  OpenAI + pgvector RAG")
    print("  DB: PostgreSQL @ %s:%s/%s" % (PG_HOST, PG_PORT, PG_DB))
    print("  LLM: %s (OpenAI)" % OPENAI_MODEL)
    print("=" * 60)

    if not test_pg_connection():
        print(
            "\n⚠️  Could not verify the 'dapchatbot' table.\n"
            "    Make sure Docker is running (docker compose up -d)\n"
            "    and embeddings/embedd.py has been executed.\n"
        )

    print("\nType your question (or 'exit' to quit):\n")
    chat_history: List[dict] = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in ("exit", "quit"):
            break

        try:
            out    = app.invoke({"question": q, "chat_history": chat_history})
            answer = out.get("answer", "(no answer)")
            print("\nAgent:", answer, "\n")
            chat_history = append_chat_history(chat_history, q, answer)
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            print(f"\n❌ Error: {exc}\n")


if __name__ == "__main__":
    run_cli()