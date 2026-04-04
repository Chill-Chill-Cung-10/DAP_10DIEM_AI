import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from pgvector.psycopg2 import register_vector
from psycopg2 import sql
from psycopg2.extras import Json

load_dotenv()
# database_url = os.getenv("DATABASE_URL", "")

# print(database_url)
EMBEDDING_DIM = 1024


def _int_or_default(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_db_candidates():

    candidates = []

    database_url = os.getenv("DATABASE_URL", "").strip()
    print(database_url)
    if database_url:
        try:
            parsed = urlparse(database_url)
            if parsed.scheme.startswith("postgres"):
                candidates.append(
                    {
                        "host": parsed.hostname or "localhost",
                        "port": parsed.port or 5432,
                        "user": parsed.username or "postgres",
                        "password": parsed.password or "",
                        "database": (parsed.path or "").lstrip("/") or "postgres",
                    }
                )
        except Exception:
            pass

    pg_host = os.getenv("PG_HOST", "").strip()
    pg_port = _int_or_default(os.getenv("PG_PORT"), 5432)
    pg_user = os.getenv("PG_USER", "").strip()
    pg_pass = os.getenv("PG_PASS", "")
    pg_db = os.getenv("PG_DB", "").strip()
    if pg_host and pg_user and pg_db:
        candidates.append(
            {
                "host": pg_host,
                "port": pg_port,
                "user": pg_user,
                "password": pg_pass,
                "database": pg_db,
            }
        )

    candidates.extend([
        {"host": "localhost", "port": 5433, "user": "postgres", "password": "postgres", "database": "demo_db"},
        {"host": "localhost", "port": 5432, "user": "postgres", "password": "postgres", "database": "demo_db"},
        {"host": "postgres", "port": 5432, "user": "postgres", "password": "postgres", "database": "demo_db"},
    ])

    deduped = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate["host"],
            candidate["port"],
            candidate["user"],
            candidate["database"],
            candidate["password"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def connect_postgres():

    candidates = build_db_candidates()

    for index, cfg in enumerate(candidates, start=1):
        try:
            print(
                f"[{index}/{len(candidates)}] host={cfg['host']} port={cfg['port']} db={cfg['database']} user={cfg['user']}"
            )
            connection = psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"],
            )

            register_vector(connection)

            print("Ket noi PostgreSQL thanh cong.")

            return connection

        except psycopg2.Error as error:

            print("Ket noi that bai:", error)

    raise RuntimeError("Khong the ket noi PostgreSQL.")


DATASET_DIR = Path("../datasets")

datasets = ["9711040808.json", "9711041707.json", "9712200299.json", "diseases.json"]


def load_datasets():
    merged_contents = []
    for file_path in DATASET_DIR.glob("*.json"):
        if file_path.name.endswith(".json.bak"):
            continue
        print(file_path.name)
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if "contents" not in data:
            raise ValueError(f"{file_path.name} missing 'contents'")

        merged_contents.extend(data["contents"])

        print(f"{file_path.name}: {len(data['contents'])} records")

    print("Total merged:", len(merged_contents))

    return {
        "contents": merged_contents
    }


def save_dataset(file_path, data):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup = file_path + f".{timestamp}.bak"
    os.rename(file_path, backup)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Backup:", backup)


def embed_vector_data(items, batch_size=16):

    texts = []
    valid_indices = []

    for idx, item in enumerate(items):

        text = item.get("vector_data")

        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
            valid_indices.append(idx)

    print("Total texts:", len(texts))

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        vectors = model.encode(batch)["dense_vecs"]

        for offset, vec in enumerate(vectors):

            item_index = valid_indices[start + offset]

            items[item_index]["embedding"] = vec.tolist()

        print(f"Embedded {start + len(batch)} / {len(texts)}")

    return items


def create_table(cursor, table):

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    query = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {table} (

        pk_id BIGSERIAL PRIMARY KEY,

        short_description TEXT,
        full_content TEXT,
        symptoms_tags JSONB,
        vector_data TEXT,

        embedding VECTOR({dim}),

        created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """).format(
        table=sql.Identifier(table),
        dim=sql.Literal(EMBEDDING_DIM)
    )

    cursor.execute(query)


def upsert_records(cursor, table, items):

    query = sql.SQL("""
    INSERT INTO {table} (
        short_description,
        full_content,
        symptoms_tags,
        vector_data,
        embedding
    )
    VALUES (%s,%s,%s,%s,%s)
    """).format(table=sql.Identifier(table))

    inserted = 0

    for item in items:

        emb = item.get("embedding")

        if not emb:
            continue

        cursor.execute(
            query,
            (
                item.get("short_description"),
                item.get("full_content"),
                Json(item.get("symptoms_tags", [])),
                item.get("vector_data"),
                emb
            )
        )

        inserted += 1

    return inserted


def main():

    # TABLE_NAME = "dap_embeddings"

    # # load dataset
    # dataset = load_datasets()
    # # print(dataset)
    # items = dataset["contents"]

    # print("records:", len(items))

    # # embedding
    # items = embed_vector_data(items)

    # connect database
    conn = connect_postgres()
    cur = conn.cursor()

    # create table nếu chưa tồn tại
    # create_table(cur, TABLE_NAME)

    # # insert dữ liệu
    # inserted = upsert_records(
    #     cur,
    #     TABLE_NAME,
    #     items
    # )

    # conn.commit()

    # print("Inserted:", inserted)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
