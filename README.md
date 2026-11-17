# Búsqueda básica de rostros

API en **Python + FastAPI** con **ArcFace (InsightFace, `buffalo_l`)** para generar embeddings
y **Chroma** como base vectorial (similitud coseno). Todo en Docker.

## Levantar
```bash
docker compose up --build
```
La primera construcción tarda unos minutos (instala dependencias y descarga el modelo, ~300 MB).
Swagger: http://localhost:8080/docs

## Servicios
- `POST /registrar` (form-data: `nombre`, `ci`, `foto`): foto con un solo rostro -> guarda embedding + nombre + ci.
- `POST /buscar` (form-data: `foto`): devuelve nombre, ci y similitud coseno del embedding más cercano.

## Probar con curl
```bash
curl -X POST localhost:8080/registrar -F nombre="Juan Perez" -F ci=1234567 -F foto=@foto1.jpg
curl -X POST localhost:8080/buscar -F foto=@foto2.jpg
```

## Notas
- Las imágenes NO se guardan: solo embeddings (512 dims) y metadatos.
- `UMBRAL_SIMILITUD` (docker-compose.yml, por defecto 0.35): bajo ese valor `/buscar` responde 404.
  Ajústalo según tus pruebas (más alto = más estricto).
- El `ci` es el ID en Chroma: no se puede registrar dos veces el mismo.
- Datos persistentes en el volumen `chroma_data` (`docker compose down -v` los borra).
- Chroma también queda expuesto en http://localhost:8001.

## Estructura
- `app/face.py`: detección y embedding (ArcFace)
- `app/db.py`: conexión a Chroma
- `app/main.py`: endpoints
