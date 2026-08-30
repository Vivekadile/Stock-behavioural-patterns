from fastapi import APIRouter, HTTPException, Query

from services import prediction_service

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

VALID_HORIZONS = [7, 30, 60, 90]


@router.get("/top")
def top_predictions(horizon: int = Query(30), limit: int = Query(10, le=48)):
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {VALID_HORIZONS}")
    return prediction_service.get_top_predictions(horizon, limit)


@router.get("/{symbol}")
def prediction_for_symbol(symbol: str, horizon: int = Query(30)):
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {VALID_HORIZONS}")
    result = prediction_service.get_prediction(symbol, horizon)
    if result is None:
        raise HTTPException(status_code=404, detail=f"'{symbol}' not found in universe")
    return result
