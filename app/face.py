"""Generación de embeddings faciales con ArcFace (InsightFace, buffalo_l). Corre local en CPU."""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

_app = None


def get_model() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


class ErrorRostro(Exception):
    pass


def _decodificar(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ErrorRostro("El archivo no es una imagen válida.")
    return img


def generar_embedding(data: bytes, exigir_un_rostro: bool) -> list[float]:
    """Devuelve el embedding (512 dims, normalizado). La imagen no se guarda en ningún lado."""
    img = _decodificar(data)
    rostros = get_model().get(img)
    if not rostros:
        raise ErrorRostro("No se detectó ningún rostro en la imagen.")
    if exigir_un_rostro and len(rostros) != 1:
        raise ErrorRostro(f"La foto debe tener exactamente un rostro (se detectaron {len(rostros)}).")
    # Si hay varios (solo en búsqueda) se usa el rostro más grande
    rostro = max(rostros, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = rostro.normed_embedding
    return [float(x) for x in emb]
