from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.v1.router import router
from app.core.config import settings
from app.db.base import init_db
from app.vision.pipeline import pipeline_registry


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield
    await pipeline_registry.shutdown()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "scai-api"}


@app.get("/monitor", include_in_schema=False)
async def monitor() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "monitor.html")
