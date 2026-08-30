from fastapi import APIRouter

from services import fyers_service

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
def health():
    return {
        "status": "ok",
        "fyers_live_data": fyers_service.is_available(),
    }
