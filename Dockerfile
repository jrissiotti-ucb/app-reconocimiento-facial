FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarga el modelo (ArcFace via InsightFace, paquete buffalo_l) al construir la imagen
RUN python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); a.prepare(ctx_id=-1)"

COPY app ./app
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
