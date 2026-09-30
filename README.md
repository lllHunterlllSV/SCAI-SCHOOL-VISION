# SCAI Backend API

MVP del backend **Smart Classroom Attention Index**. Expone sesiones, métricas agregadas,
eventos, reportes, feedback, recomendaciones y actualizaciones WebSocket. El modo demo genera
features colectivas sintéticas para integrar el frontend sin cámara ni modelos pesados.

## Arranque local

Todo el proyecto usa exclusivamente **Python 3.12**. Esta versión es compartida por el entorno
local, Docker, OpenCV, YOLO y MediaPipe.

```bash
py -3.12 -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,vision]"
python -m uvicorn app.main:app --reload
```

Swagger queda disponible en `http://localhost:8000/docs`. También se puede iniciar con
`docker compose up --build`.

## Flujo rápido para el frontend

1. Crear sesión con `POST /api/v1/sessions`:

```json
{
  "class_id": "calculo-2-a",
  "mode": "lecture",
  "video_source_type": "file_demo",
  "source_config": {"synthetic": true},
  "consent_confirmed": true
}
```

2. Conectar a `ws://localhost:8000/api/v1/ws/sessions/{id}`.
3. Consultar `GET /api/v1/sessions/{id}/scai` o `/report`.
4. Finalizar con `PATCH /api/v1/sessions/{id}/stop`.

En desarrollo la autenticación es opcional. En otros entornos se obtiene un JWT con
`POST /api/v1/auth/login` y se usa `Authorization: Bearer <token>`; el WebSocket acepta
`?token=<token>`. El login incluido es únicamente una costura de MVP y debe reemplazarse por
OIDC institucional antes de producción.

## Privacidad y uso responsable

- El esquema no contiene frames, imágenes, video, embeddings faciales ni identificadores de
  estudiantes. El pipeline recibe y persiste exclusivamente features colectivas agregadas.
- Crear una sesión exige `consent_confirmed=true`.
- Los eventos son señales probabilísticas para revisión humana. No son evidencia concluyente ni
  deben usarse para sanciones automáticas.
- `DELETE /api/v1/sessions/{id}` elimina en cascada métricas, eventos, recomendaciones y feedback.

## Visión real: OpenCV + YOLO + MediaPipe

El pipeline visual se instala en el mismo entorno Python 3.12 del backend:

```bash
pip install -e ".[dev,vision]"
```

Ultralytics descarga `yolo11n.pt` durante el primer uso si no está localmente. Para una cámara:

```json
{
  "class_id": "calculo-2-a",
  "mode": "lecture",
  "video_source_type": "live",
  "source_config": {"device_index": 0, "sampling_fps": 5, "device": "cpu"},
  "consent_confirmed": true
}
```

Para video usa `"video_source_type": "file_demo"` y
`"source_config": {"file_path": "C:/ruta/demo.mp4"}`. También acepta `model_path`,
`confidence` y `device` (`cpu`, `0`, etc.). En modo `exam`, una detección `cell phone` genera
una señal anónima que requiere revisión humana.

OpenCV entrega cada frame en memoria; YOLO produce cajas numéricas y MediaPipe procesa recortes
transitorios. Tras extraer los agregados se elimina la referencia al frame. Frames, recortes,
landmarks y cajas individuales nunca se escriben en la base de datos ni en archivos.

## Verificación

```bash
pytest
ruff check .
```
