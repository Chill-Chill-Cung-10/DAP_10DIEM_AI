import os
import json
from urllib.parse import urlparse
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from FlagEmbedding import BGEM3FlagModel

load_dotenv()


def _int_or_default(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_db_candidates():
    """Tạo danh sách cấu hình DB để thử kết nối theo thứ tự ưu tiên."""
    candidates = []

    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        try:
            parsed = urlparse(database_url)
            if parsed.scheme.startswith("postgres"):
                candidates.append({
                    "host": parsed.hostname or "localhost",
                    "port": parsed.port or 5432,
                    "user": parsed.username or "postgres",
                    "password": parsed.password or "",
                    "database": (parsed.path or "").lstrip("/") or "postgres"
                })
        except Exception:
            pass

    pg_host = os.getenv("PG_HOST", "").strip()
    pg_port = _int_or_default(os.getenv("PG_PORT"), 5432)
    pg_user = os.getenv("PG_USER", "").strip()
    pg_pass = os.getenv("PG_PASS", "")
    pg_db = os.getenv("PG_DB", "").strip()

    if pg_host and pg_user and pg_db:
        candidates.append({
            "host": pg_host,
            "port": pg_port,
            "user": pg_user,
            "password": pg_pass,
            "database": pg_db
        })

    # Fallback cho các tình huống phổ biến
    candidates.extend([
        {"host": "localhost", "port": 5433, "user": "postgres", "password": "postgres", "database": "demo_db"},
        {"host": "localhost", "port": 5432, "user": "postgres", "password": "postgres", "database": "demo_db"},
        {"host": "postgres", "port": 5432, "user": "postgres", "password": "postgres", "database": "demo_db"},
    ])

    # Loại trùng theo bộ khóa kết nối
    deduped = []
    seen = set()
    for c in candidates:
        key = (c["host"], c["port"], c["user"], c["database"], c["password"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _connect_postgres():
    candidates = _build_db_candidates()
    last_error = None

    print(f"\n{'='*50}")
    print("🔌 Đang thử kết nối PostgreSQL...")
    print(f"{'='*50}")

    for i, cfg in enumerate(candidates, start=1):
        try:
            print(f"[{i}/{len(candidates)}] host={cfg['host']} port={cfg['port']} db={cfg['database']} user={cfg['user']}")
            conn = psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"]
            )
            print("✅ Kết nối PostgreSQL thành công")
            return conn
        except psycopg2.Error as e:
            last_error = e
            print(f"⚠️ Kết nối thất bại: {e.pgerror or str(e).splitlines()[0]}")

    raise last_error if last_error else RuntimeError("Không thể kết nối PostgreSQL")


def _pick_first_text(values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _records_from_items(items, source_name):
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue

        question = _pick_first_text([
            item.get("summary"),
            item.get("question"),
            item.get("title"),
            item.get("heading")
        ])
        answer = _pick_first_text([
            item.get("full_text"),
            item.get("answer"),
            item.get("text"),
            item.get("description")
        ])

        if question:
            if not answer:
                answer = question
            records.append({
                "question": question,
                "answer": answer,
                "source": source_name
            })
    return records


def load_records_from_datasets(datasets_dir):
    records = []
    json_files = sorted(
        [
            f for f in os.listdir(datasets_dir)
            if f.lower().endswith(".json") and not f.lower().endswith(".json.bak")
        ]
    )

    print(f"\n{'='*50}")
    print("📂 Đang đọc JSON từ datasets...")
    print(f"{'='*50}")

    for file_name in json_files:
        file_path = os.path.join(datasets_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_records = []
            if isinstance(data, dict):
                if isinstance(data.get("contents"), list):
                    file_records.extend(_records_from_items(data["contents"], file_name))
                else:
                    file_records.extend(_records_from_items([data], file_name))
            elif isinstance(data, list):
                normalized_items = []
                for entry in data:
                    if isinstance(entry, dict) and isinstance(entry.get("content"), dict):
                        normalized_items.append(entry["content"])
                    elif isinstance(entry, dict):
                        normalized_items.append(entry)
                file_records.extend(_records_from_items(normalized_items, file_name))

            records.extend(file_records)
            print(f"✅ {file_name}: {len(file_records)} records")
        except Exception as e:
            print(f"❌ Lỗi đọc {file_name}: {e}")

    print(f"📊 Tổng records từ JSON: {len(records)}")
    return records

try:
    connection = _connect_postgres()
    cursor = connection.cursor()
    
    # Tạo extension pgvector nếu chưa có
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Tạo bảng diseases
    print(f"\n{'='*50}")
    print("🔧 Tạo bảng diseases...")
    print(f"{'='*50}")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diseases (
            id SERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            embedding vector(1024)
        )
    """)
    print("✅ Đã tạo bảng diseases")
    
    # Đọc dữ liệu từ datasets JSON
    base_dir = os.path.dirname(os.path.dirname(__file__))
    datasets_dir = os.path.join(base_dir, "datasets")
    records = load_records_from_datasets(datasets_dir)

    # Fallback sang CSV nếu JSON không có dữ liệu hợp lệ
    if not records:
        print(f"\n{'='*50}")
        print("📂 Không có dữ liệu JSON hợp lệ, chuyển sang đọc embeddings.csv...")
        print(f"{'='*50}")

        csv_path = os.path.join(os.path.dirname(__file__), "embeddings.csv")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["question"])

        for _, row in df.iterrows():
            question = row.get("question")
            if isinstance(question, str) and question.strip():
                records.append({
                    "question": question.strip(),
                    "answer": str(row.get("answer", "") or ""),
                    "source": "embeddings.csv"
                })

    print(f"✅ Số records dùng để embed: {len(records)}")
    
    # Khởi tạo BGE-M3 embedding model
    print(f"\n🤖 Khởi tạo BGE-M3 Embedding Model...")
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # Lưu cặp (index, vector) để mapping đúng
    successful_embeddings = []
    
    print(f"\n{'='*50}")
    print("🔄 Bắt đầu embedding...")
    print(f"{'='*50}\n")
    
    for idx, row in enumerate(records):
        question = row["question"]
        
        if isinstance(question, str) and question.strip():
            try: 
                # BGE-M3 trả về dict, lấy 'dense_vecs'
                embeddings = embedding_model.encode([question])
                vector = embeddings['dense_vecs'][0].tolist()
                successful_embeddings.append((idx, vector))
                print(f"✅ Embedded {idx+1}/{len(records)}: {question[:50]}...")
            except Exception as e:
                print(f"❌ Lỗi embedding dòng {idx+1}: {e}")
        else:
            print(f"⚠️ Bỏ qua dòng {idx+1}: question không hợp lệ")
    
    # Insert vào database
    print(f"\n{'='*50}")
    print("💾 Đang lưu vào database...")
    print(f"{'='*50}\n")
    
    inserted_count = 0
    for idx, vector in successful_embeddings:
        row = records[idx]
        try:
            cursor.execute(
                """INSERT INTO diseases(
                    question, answer, embedding
                ) VALUES (%s, %s, %s)""",
                (
                    row["question"],
                    row.get("answer", ""),
                    vector
                )
            )
            inserted_count += 1
            print(f"✅ Đã lưu dòng {idx+1}: {row['question'][:50]}...")
        except Exception as e:
            print(f"❌ Lỗi insert dòng {idx+1}: {e}")
            print(f"   Data: {row}")
    
    connection.commit()
    print(f"\n{'='*50}")
    print(f"🎉 Hoàn thành! Đã lưu {inserted_count}/{len(successful_embeddings)} dòng vào database")
    print(f"{'='*50}")
    
    print(f"\n{'='*50}")
    print("🎉 Hoàn thành tất cả!")
    print(f"{'='*50}")

except psycopg2.Error as e:
    print("❌ Lỗi kết nối PostgreSQL:")
    print(e)
except Exception as e:
    print("❌ Lỗi khác:")
    print(e)
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals():
        connection.close()
    print("\n✅ Đã đóng kết nối database")