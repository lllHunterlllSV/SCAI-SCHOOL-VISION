from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def create_access_token(subject: str, role: str = "teacher") -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


async def current_user(token: str | None = Depends(oauth2_scheme)) -> dict[str, str]:
    # Development is frictionless; production always requires a valid token.
    if not token and settings.environment == "development":
        return {"sub": "local-teacher", "role": "teacher"}
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return {"sub": str(payload["sub"]), "role": str(payload.get("role", "teacher"))}
    except (jwt.InvalidTokenError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


def validate_ws_token(token: str | None) -> bool:
    if not token and settings.environment == "development":
        return True
    if not token:
        return False
    try:
        jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return True
    except jwt.InvalidTokenError:
        return False
