from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..dependencies import CurrentUser

router = APIRouter(tags=["health"])


@router.get("/health", response_class=JSONResponse)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/protected", response_class=JSONResponse)
async def health_protected(current_user: CurrentUser) -> dict[str, str]:
    return {"status": "ok", "user": current_user.email}
