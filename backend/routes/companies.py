from fastapi import APIRouter, HTTPException, Query

from services import prediction_service, history_service

router = APIRouter(prefix="/api/companies", tags=["companies"])


@router.get("")
def list_companies():
    return prediction_service.list_companies()


@router.get("/{symbol}")
def get_company(symbol: str):
    result = prediction_service.get_company_detail(symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"'{symbol}' not found in universe")
    return result


@router.get("/{symbol}/history")
def get_company_history(symbol: str, days: int = Query(180, le=1000)):
    result = history_service.get_history(symbol, days)
    if result is None:
        raise HTTPException(status_code=404, detail=f"'{symbol}' not found in universe")
    return result
