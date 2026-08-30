from fastapi import APIRouter

from services import market_service

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/overview")
def market_overview():
    return market_service.get_market_overview()
