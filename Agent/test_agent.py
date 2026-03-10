"""Quick smoke-test for Agent.py ↔ PostgreSQL + pgvector pipeline."""
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "postgres")
PG_DB   = os.getenv("PG_DB", "demo_db")

print("=" * 60)
print("  Agent.py Smoke Test")
print("=" * 60)

# 1. PostgreSQL connection
print("\n[1/4] Testing PostgreSQL connection...")
conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, database=PG_DB)
cur = conn.cursor()
cur.execute("SELECT 1")
print("  ✅ PostgreSQL connected")

# 2. Table exists
print("\n[2/4] Checking 'dapchatbot' table...")
cur.execute("SELECT COUNT(*) FROM dapchatbot")
count = cur.fetchone()[0]
print(f"  ✅ Table 'dapchatbot' exists with {count} rows")

# 3. pgvector extension
print("\n[3/4] Checking pgvector extension...")
cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
row = cur.fetchone()
print(f"  ✅ pgvector v{row[0]} installed")

# 4. BGE-M3 + similarity search
print("\n[4/4] Testing BGE-M3 embedding + pgvector search...")
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
query = "Quy định về ly hôn"
vec = model.encode([query])["dense_vecs"][0].tolist()

cur2 = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cur2.execute(
    """
    SELECT id, LEFT(question, 80) AS q, 1 - (embedding <=> %s::vector) AS sim
    FROM dapchatbot
    ORDER BY embedding <=> %s::vector
    LIMIT 3
    """,
    (str(vec), str(vec)),
)
results = cur2.fetchall()
for r in results:
    print(f"  📄 id={r['id']}  sim={r['sim']:.4f}  {r['q']}")

cur2.close()
cur.close()
conn.close()

print("\n" + "=" * 60)
print("  ✅ All tests passed! Agent.py is ready to run.")
print("=" * 60)
