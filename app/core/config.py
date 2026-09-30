from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SCAI Backend API"
    environment: str = "development"
    secret_key: str = "development-only-change-me"
    database_url: str = "sqlite+aiosqlite:///./scai.db"
    allowed_origins: list[str] = ["http://localhost:3000"]
    access_token_minutes: int = 60
    demo_tick_seconds: float = 1.0
    demo_max_ticks: int = 0
    default_sampling_fps: float = 5.0
    yolo_model_path: str = "yolo11n.pt"
    yolo_device: str = "cpu"
    yolo_confidence: float = 0.35
    test_video_path: str = (
        "C:/Users/edwin/Downloads/Zoom Class Meeting Downing Soc 220 "
        "1\u204428\u20442021/TEST.mp4"
    )
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
