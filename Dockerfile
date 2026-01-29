FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080

# (optional) system deps that often help wheels build/install (lightgbm can need this)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install deps first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# install CPU-only torch (avoids CUDA 8GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# copy app code
COPY . .

# make src importable
ENV PYTHONPATH=/app/src

CMD ["python","-m","uvicorn","src.client.app:app","--host","0.0.0.0","--port","8080"]