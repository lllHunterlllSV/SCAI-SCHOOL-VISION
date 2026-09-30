import asyncio

from app.db.schemas import Thresholds


class ThresholdStore:
    def __init__(self) -> None:
        self._value = Thresholds()
        self._lock = asyncio.Lock()

    async def get(self) -> Thresholds:
        return self._value.model_copy()

    async def set(self, value: Thresholds) -> Thresholds:
        async with self._lock:
            self._value = value
        return value.model_copy()


threshold_store = ThresholdStore()
