FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

EXPOSE 3400

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3400"]