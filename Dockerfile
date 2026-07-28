FROM python:3.11-slim

# dependencies sistem yang dibutuhkan opencv & mediapipe
RUN apt-get update && apt-get install -y \
    libgl1\
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD gunicorn -w 1 -b 0.0.0.0:${PORT:-8000} --timeout 120 app:app