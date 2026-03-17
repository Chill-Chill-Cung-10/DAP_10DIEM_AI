"""CLI entrypoint with INFO-level logging enabled."""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

from Agent import app, logger, OPENAI_MODEL, PG_DB, PG_HOST, PG_PORT, test_pg_connection, append_chat_history  # noqa: E402

logger.setLevel(logging.INFO)


def run_cli_log_mode() -> None:
    print("=" * 60)
    print("  LangGraph Agent (LOG MODE)")
    print("  DB: PostgreSQL @ %s:%s/%s" % (PG_HOST, PG_PORT, PG_DB))
    print("  LLM: %s (OpenAI)" % OPENAI_MODEL)
    print("=" * 60)

    if not test_pg_connection():
        print(
            "\nWARNING: Could not verify the 'dap_embeddings' table.\n"
            "    Make sure Docker is running (docker compose up -d)\n"
            "    and embeddings/embedd.py has been executed.\n"
        )

    print("\nType your question (or 'exit' to quit):\n")
    chat_history = []

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
            logger.error("Error during invoke: %s", exc, exc_info=True)
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    run_cli_log_mode()