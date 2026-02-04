# 1. Sử dụng Python 3.11 slim - Nhẹ và ổn định
FROM python:3.11-slim

# 2. Biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Cài gói hệ thống cần thiết
# Thêm 'curl' để debug kết nối với LM Studio (host.docker.internal)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Cài đặt Dependencies 
# Tách riêng phần này để tận dụng Docker Cache (chỉ cài lại khi file requirements thay đổi)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ mã nguồn vào container
COPY . .

# 7. Mở port 3000 (Khớp với docker-compose.yml port: 3000:3000)
EXPOSE 3000

# 8. Lệnh chạy ứng dụng
# Sử dụng python trực tiếp (vì đã cài trong môi trường slim)
CMD ["python", "main.py"]