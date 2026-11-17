import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import db, face

UMBRAL = float(os.getenv("UMBRAL_SIMILITUD", "0.35"))
estado = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    face.get_model()                    # carga el modelo una sola vez
    estado["coleccion"] = db.conectar()
    yield


app = FastAPI(
    title="Búsqueda básica de rostros",
    description="API con ArcFace (InsightFace) + Chroma. Servicios: Registrar y Buscar.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/registrar", summary="Registrar una persona a partir de una foto con un solo rostro")
async def registrar(
    nombre: str = Form(..., description="Nombre de la persona"),
    ci: str = Form(..., description="Cédula de identidad (se usa como ID único)"),
    foto: UploadFile = File(..., description="Foto con un solo rostro"),
):
    col = estado["coleccion"]
    if col.get(ids=[ci])["ids"]:
        raise HTTPException(409, f"Ya existe una persona registrada con ci {ci}.")
    try:
        emb = face.generar_embedding(await foto.read(), exigir_un_rostro=True)
    except face.ErrorRostro as e:
        raise HTTPException(400, str(e))
    # Solo se guardan el embedding y los datos de la persona (no la imagen)
    col.add(ids=[ci], embeddings=[emb], metadatas=[{"nombre": nombre, "ci": ci}])
    return {"mensaje": "Persona registrada", "nombre": nombre, "ci": ci}


@app.post("/buscar", summary="Buscar la persona más parecida a la foto")
async def buscar(foto: UploadFile = File(..., description="Foto a buscar")):
    col = estado["coleccion"]
    if col.count() == 0:
        raise HTTPException(404, "No hay personas registradas.")
    try:
        emb = face.generar_embedding(await foto.read(), exigir_un_rostro=False)
    except face.ErrorRostro as e:
        raise HTTPException(400, str(e))
    res = col.query(query_embeddings=[emb], n_results=1, include=["metadatas", "distances"])
    meta = res["metadatas"][0][0]
    similitud = 1.0 - res["distances"][0][0]  # distancia coseno -> similitud coseno
    if similitud < UMBRAL:
        raise HTTPException(
            404,
            f"No hay coincidencia suficiente (similitud {similitud:.3f} < umbral {UMBRAL}).",
        )
    return {"nombre": meta["nombre"], "ci": meta["ci"], "similitud_coseno": round(similitud, 4)}
