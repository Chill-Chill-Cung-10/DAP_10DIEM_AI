"""
Agent_log.py — Version with full logging enabled.

Same agent as Agent.py but with INFO-level logs visible in console.
Use this for debugging / monitoring the agent pipeline.

Usage:
    from Agent.Agent_log import app   # logs visible
    # vs
    from Agent.Agent import app       # logs suppressed
"""

import logging

# Enable INFO logging BEFORE importing Agent (so logger picks it up)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,  # override Agent.py's WARNING-level config
)

from Agent import (  # noqa: F401, E402
    app,
    AgentState,
    test_pg_connection,
    pgvector_search,
    PG_HOST,
    PG_PORT,
    PG_DB,
    OPENAI_MODEL,
    logger,
)

# Ensure the agent logger is set to INFO
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    print("=" * 60)
    print("  LangGraph Agent (LOG MODE)")
    print("  DB: PostgreSQL @ %s:%s/%s" % (PG_HOST, PG_PORT, PG_DB))
    print("  LLM: %s (OpenAI)" % OPENAI_MODEL)
    print("=" * 60)

    if not test_pg_connection():
        print(
            "\n⚠️  Could not verify the 'diseases' table.\n"
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
            out = app.invoke({"question": q, "chat_history": chat_history})
            answer = out.get("answer", "(no answer)")
            print("\nAgent:", answer, "\n")
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": answer})
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
        except Exception as exc:
            logger.error("Error during invoke: %s", exc, exc_info=True)
            print(f"\n❌ Error: {exc}\n")
