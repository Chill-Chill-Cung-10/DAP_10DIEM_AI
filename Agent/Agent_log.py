"""
Agent_log.py — Thin compatibility wrapper.

This file previously contained a separate copy of the agent with logging.
Agent.py now includes full structured logging, so this module simply
re-exports everything from Agent.py.

Usage:
    from Agent.Agent_log import app   # works the same as before
"""

from Agent.Agent import (  # noqa: F401
    app,
    AgentState,
    test_pg_connection,
    pgvector_search,
    PG_HOST,
    PG_PORT,
    PG_DB,
    LMSTUDIO_MODEL,
    LMSTUDIO_BASE_URL,
    logger,
)

if __name__ == "__main__":
    # Delegate to the main agent entry point
    from Agent.Agent import (
        test_pg_connection as _check,
        app as _app,
    )
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  LangGraph Agent (via Agent_log wrapper)")
    print("  DB: PostgreSQL @ %s:%s/%s" % (PG_HOST, PG_PORT, PG_DB))
    print("  LLM: %s @ %s" % (LMSTUDIO_MODEL, LMSTUDIO_BASE_URL))
    print("=" * 60)

    if not _check():
        print(
            "\n⚠️  Could not verify the 'dapchatbot' table.\n"
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
            out = _app.invoke({"question": q, "chat_history": chat_history})
            answer = out.get("answer", "(no answer)")
            print("\nAgent:", answer, "\n")
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": answer})
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
        except Exception as exc:
            logger.error("Error during invoke: %s", exc, exc_info=True)
            print(f"\n❌ Error: {exc}\n")
