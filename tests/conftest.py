import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_scai.db"
os.environ["ENVIRONMENT"] = "development"
os.environ["DEMO_TICK_SECONDS"] = "0.02"

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value,
    ):
        yield value
