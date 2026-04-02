from fastapi import APIRouter, Request, HTTPException
from api.schemas import PredictRequest, PredictResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    try:
        service = request.app.state.predict_service
        response = service.predict(body.text)
        return PredictResponse(predictions=response, text_received=body.text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))