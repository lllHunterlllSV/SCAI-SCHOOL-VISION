import asyncio


class InMemoryPreviewHub:
    """Holds only the latest encoded preview; nothing is written to persistent storage."""

    def __init__(self) -> None:
        self._latest: dict[str, tuple[int, bytes]] = {}
        self._sequence = 0

    def publish(self, session_id: str, jpeg: bytes) -> None:
        self._sequence += 1
        self._latest[session_id] = (self._sequence, jpeg)

    def remove(self, session_id: str) -> None:
        self._latest.pop(session_id, None)

    async def stream(self, session_id: str):
        last_sequence = -1
        try:
            while True:
                item = self._latest.get(session_id)
                if item and item[0] != last_sequence:
                    last_sequence, jpeg = item
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.08)
        finally:
            # The pipeline owns cleanup; disconnecting one viewer must not affect others.
            pass


preview_hub = InMemoryPreviewHub()
