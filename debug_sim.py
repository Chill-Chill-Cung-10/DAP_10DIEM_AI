"""Debug: simulate what pgvector_search returns for the user's exact query."""
import psycopg2
import psycopg2.extras

conn = psycopg2.connect(host="localhost", port=5433, user="postgres", password="postgres", database="demo_db")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Get the stored embedding for the matching question
# Then simulate pgvector_search using that embedding as if it were the runtime embedding
# (best case scenario - runtime BGE-M3 produces same vector)
user_query = "Quy định pháp luật về điều kiện kết hôn tại Việt Nam là gì?"

cur.execute("""
    WITH target AS (
        SELECT embedding
        FROM diseases
        WHERE question ILIKE '%%điều kiện kết hôn%%'
        LIMIT 1
    )
    SELECT d.question,
           1 - (d.embedding <=> t.embedding) AS vec_sim,
           ts_rank(
               to_tsvector('simple', d.question || ' ' || d.answer),
               plainto_tsquery('simple', %s)
           ) AS text_rank
    FROM diseases d, target t
    WHERE 1 - (d.embedding <=> t.embedding) >= 0.45
    ORDER BY (
        0.7 * (1 - (d.embedding <=> t.embedding))
        + 0.3 * ts_rank(
            to_tsvector('simple', d.question || ' ' || d.answer),
            plainto_tsquery('simple', %s)
        )
    ) DESC
    LIMIT 5
""", (user_query, user_query))

print("User query:", user_query)
print()
print("=== pgvector_search results (simulated) ===")
for r in cur.fetchall():
    vec = float(r["vec_sim"])
    txt = float(r["text_rank"])
    hybrid = 0.7 * vec + 0.3 * txt
    print("HYBRID=%.4f  vec=%.4f  txt=%.6f | %s" % (
        hybrid, vec, txt, r["question"][:80]
    ))
    print("  → Would pass 0.55 threshold?", "YES" if hybrid >= 0.55 else "NO")
    print("  → Would pass 0.45 pgvector filter?", "YES" if vec >= 0.45 else "NO")

# Also check: how many total rows?
cur.execute("SELECT COUNT(*) FROM diseases")
print("\nTotal rows in diseases:", cur.fetchone()[0])

# Check embedding dimensions
cur.execute("SELECT vector_dims(embedding) FROM diseases LIMIT 1")
print("Embedding dimensions:", cur.fetchone()[0])

cur.close()
conn.close()
