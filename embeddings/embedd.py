import argparse
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

EMBEDDING_DIM = 1024


def _int_or_default(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_db_candidates():
    candidates = []

    database_url = os.getenv("DATABASE_URL", "").strip()
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

    candidates.extend(
        [
            {"host": "localhost", "port": 5433, "user": "postgres", "password": "postgres", "database": "demo_db"},
            {"host": "localhost", "port": 5432, "user": "postgres", "password": "postgres", "database": "demo_db"},
            {"host": "postgres", "port": 5432, "user": "postgres", "password": "postgres", "database": "demo_db"},
        ]
    )

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


def _connect_postgres():
    candidates = _build_db_candidates()
    last_error = None

    print("\n" + "=" * 60)
    print("Dang thu ket noi PostgreSQL...")
    print("=" * 60)

    for index, cfg in enumerate(candidates, start=1):
        try:
            print(
                f"[{index}/{len(candidates)}] host={cfg['host']} "
                f"port={cfg['port']} db={cfg['database']} user={cfg['user']}"
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
            last_error = error
            print(f"Ket noi that bai: {error.pgerror or str(error).splitlines()[0]}")

    raise last_error if last_error else RuntimeError("Khong the ket noi PostgreSQL.")


def _list_dataset_files(datasets_dir):
    return sorted(
        [path for path in datasets_dir.glob("*.json") if not path.name.lower().endswith(".json.bak")]
    )


def _choose_dataset_file(datasets_dir, explicit_file=None):
    if explicit_file:
        file_path = Path(explicit_file)
        if not file_path.is_absolute():
            file_path = (Path.cwd() / explicit_file).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Khong tim thay file: {file_path}")
        return file_path

    files = _list_dataset_files(datasets_dir)
    if not files:
        raise FileNotFoundError(f"Khong co file JSON trong: {datasets_dir}")

    print("\n" + "=" * 60)
    print("Danh sach file datasets:")
    print("=" * 60)
    for index, path in enumerate(files, start=1):
        print(f"{index}. {path.name}")

    while True:
        picked = input("Chon so thu tu file can embedding: ").strip()
        try:
            selected_index = int(picked)
            if 1 <= selected_index <= len(files):
                return files[selected_index - 1]
        except ValueError:
            pass
        print("Lua chon khong hop le, vui long nhap lai.")


def _load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or "contents" not in data or not isinstance(data["contents"], list):
        raise ValueError("File JSON phai co format: {'contents': [...]} ")
    return data


def _save_dataset(file_path, data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(file_path.suffix + f".{timestamp}.bak")
    file_path.replace(backup_path)

    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    print(f"Da sao luu file goc: {backup_path}")
    print(f"Da cap nhat file JSON: {file_path}")


def _embed_vector_data(items, batch_size):
    texts = []
    valid_indices = []

    for index, item in enumerate(items):
        text = item.get("vector_data", "")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
            valid_indices.append(index)

    if not texts:
        return 0

    print("\n" + "=" * 60)
    print("Khoi tao model BGE-M3 va bat dau embedding...")
    print("=" * 60)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    embedded_count = 0
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_vectors = model.encode(batch_texts)["dense_vecs"]

        for offset, vector in enumerate(batch_vectors):
            item_index = valid_indices[start + offset]
            items[item_index]["embedding"] = vector.tolist()
            embedded_count += 1

        print(f"Embedded {end}/{len(texts)} records")

    return embedded_count


def _create_table(cursor, table_name):
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    create_table_query = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table_name} (
            pk_id BIGSERIAL PRIMARY KEY,
            item_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            vector_data TEXT NOT NULL,
            display_title TEXT,
            short_description TEXT,
            full_content TEXT,
            symptoms_tags JSONB,
            embedding vector({dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (source_file, item_id)
        )
        """
    ).format(table_name=sql.Identifier(table_name), dim=sql.Literal(EMBEDDING_DIM))
    cursor.execute(create_table_query)


def _upsert_records(cursor, table_name, source_file, items):
    upsert_query = sql.SQL(
        """
        INSERT INTO {table_name} (
            item_id,
            source_file,
            vector_data,
            display_title,
            short_description,
            full_content,
            symptoms_tags,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_file, item_id)
        DO UPDATE SET
            vector_data = EXCLUDED.vector_data,
            display_title = EXCLUDED.display_title,
            short_description = EXCLUDED.short_description,
            full_content = EXCLUDED.full_content,
            symptoms_tags = EXCLUDED.symptoms_tags,
            embedding = EXCLUDED.embedding
        """
    ).format(table_name=sql.Identifier(table_name))

    inserted = 0
    for item in items:
        vector = item.get("embedding")
        vector_data = item.get("vector_data")
        item_id = item.get("id")
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
            continue
        if not isinstance(vector_data, str) or not vector_data.strip():
            continue
        if not isinstance(item_id, str) or not item_id.strip():
            continue

        cursor.execute(
            upsert_query,
            (
                item_id.strip(),
                source_file,
                vector_data.strip(),
                item.get("display_title"),
                item.get("short_description"),
                item.get("full_content"),
                Json(item.get("symptoms_tags", [])),
                vector,
            ),
        )
        inserted += 1

    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline embedding: chon file JSON -> embed vector_data -> ghi embedding vao JSON -> upsert vao PostgreSQL."
    )
    parser.add_argument("--file", help="Duong dan file JSON can embedding. Neu bo trong se cho chon tu datasets/.")
    parser.add_argument("--table", default="dataset_embeddings", help="Ten bang PostgreSQL, mac dinh: dataset_embeddings")
    parser.add_argument("--batch-size", type=int, default=16, help="So luong ban ghi embedding moi batch.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    datasets_dir = base_dir / "datasets"
    file_path = _choose_dataset_file(datasets_dir, explicit_file=args.file)

    print("\n" + "=" * 60)
    print(f"File duoc chon: {file_path}")
    print("=" * 60)

    data = _load_dataset(file_path)
    items = data["contents"]
    print(f"So records tim thay trong file: {len(items)}")

    embedded_count = _embed_vector_data(items, batch_size=max(1, args.batch_size))
    if embedded_count == 0:
        raise ValueError("Khong co record hop le de embedding trong truong 'vector_data'.")
    print(f"So records da embed: {embedded_count}")

    _save_dataset(file_path, data)

    connection = None
    cursor = None
    try:
        connection = _connect_postgres()
        cursor = connection.cursor()

        print("\n" + "=" * 60)
        print(f"Tao bang neu chua ton tai: {args.table}")
        print("=" * 60)
        _create_table(cursor, args.table)

        print("\n" + "=" * 60)
        print("Dang upsert du lieu vao database...")
        print("=" * 60)
        inserted = _upsert_records(cursor, args.table, file_path.name, items)
        connection.commit()
        print(f"Upsert thanh cong {inserted} records vao bang '{args.table}'.")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            print("Da dong ket noi database.")


if __name__ == "__main__":
    main()
