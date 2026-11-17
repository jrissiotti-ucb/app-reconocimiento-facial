"""Conexión a Chroma (contenedor bd-chroma)."""
import os
import time
import chromadb

COLECCION = "rostros"


def conectar():
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    ultimo = None
    for _ in range(30):  # espera a que Chroma termine de arrancar
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            return client.get_or_create_collection(
                name=COLECCION, metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            ultimo = e
            time.sleep(2)
    raise RuntimeError(f"No se pudo conectar a Chroma en {host}:{port}: {ultimo}")
