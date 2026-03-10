import re
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# Sử dụng thông tin từ docker-compose
PG_HOST_AI = "localhost"  # hoặc "postgres" nếu chạy trong Docker
PG_PORT_AI = 5432
PG_USER = "postgres"  # từ docker-compose.yml
PG_PASS = "postgres"  # từ docker-compose.yml
PG_DB = "demo_db"    # từ docker-compose.yml

try:
    connection = psycopg2.connect(
        host=PG_HOST_AI,
        port=PG_PORT_AI,
        user=PG_USER,
        password=PG_PASS,
        database=PG_DB
    )
    cursor = connection.cursor()
    
    # Tạo extension pgvector nếu chưa có
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Tạo bảng dapchatbot
    print(f"\n{'='*50}")
    print("🔧 Tạo bảng dapchatbot...")
    print(f"{'='*50}")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dapchatbot (
            id SERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            embedding vector(1024)
        )
    """)
    print("✅ Đã tạo bảng dapchatbot")
    
    # Đọc CSV file
    print(f"\n{'='*50}")
    print("📂 Đang đọc file embeddings.csv...")
    print(f"{'='*50}")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'embeddings.csv')
    df = pd.read_csv(csv_path)
    
    # Hiển thị thông tin
    print(f"✅ Đã đọc {len(df)} dòng từ CSV")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Loại bỏ các dòng có question rỗng
    df = df.dropna(subset=['question'])
    print(f"📊 Sau khi loại bỏ question rỗng: {len(df)} dòng")
    
    # Khởi tạo BGE-M3 embedding model
    print(f"\n🤖 Khởi tạo BGE-M3 Embedding Model...")
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # Lưu cặp (index, vector) để mapping đúng
    successful_embeddings = []
    
    print(f"\n{'='*50}")
    print("🔄 Bắt đầu embedding...")
    print(f"{'='*50}\n")
    
    for idx, row in df.iterrows():
        question = row['question']
        
        if isinstance(question, str) and question.strip():
            try: 
                # BGE-M3 trả về dict, lấy 'dense_vecs'
                embeddings = embedding_model.encode([question])
                vector = embeddings['dense_vecs'][0].tolist()
                successful_embeddings.append((idx, vector))
                print(f"✅ Embedded {idx+1}/{len(df)}: {question[:50]}...")
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
        row = df.iloc[idx]
        try:
            cursor.execute(
                """INSERT INTO dapchatbot(
                    question, answer, embedding
                ) VALUES (%s, %s, %s)""",
                (
                    row['question'],
                    row.get('answer', ''),
                    vector
                )
            )
            inserted_count += 1
            print(f"✅ Đã lưu dòng {idx+1}: {row['question'][:50]}...")
        except Exception as e:
            print(f"❌ Lỗi insert dòng {idx+1}: {e}")
            print(f"   Data: {row.to_dict()}")
    
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