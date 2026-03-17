agent/
├── __init__.py              ← Public API (app, logger, helpers)
├── config.py                ← Tất cả env vars & constants
│
├── db/
│   ├── pool.py              ← Connection pool (get/put)
│   ├── health.py            ← test_pg_connection + HNSW bootstrap
│   └── search.py            ← pgvector_search, exact_db_lookup, Document
│
├── embedding/
│   └── model.py             ← BGE-M3 + LRU cache
│
├── graph/
│   ├── state.py             ← AgentState TypedDict
│   └── builder.py           ← build_app() — wiring nodes & edges
│
├── nodes/
│   ├── router.py            ← intent_router (keyword-based)
│   ├── simple.py            ← greeting, system_info, clarify, answer
│   ├── retrieval.py         ← direct_answer, query_rewriter, rag, topic_judge
│   └── synthesis.py         ← answer_draft, verifier, evidence pipeline
│
├── routing/
│   └── edges.py             ← route_intent, route_direct_quality, ...
│
└── utils/
    ├── text.py              ← normalize, canonicalize, strip_accents
    └── llm.py               ← llm client, safe_invoke, docs_context, chat_history

main.py                      ← CLI (quiet logging)
main_log.py                  ← CLI (INFO logging, thay Agent_log.py)