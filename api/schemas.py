from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")

class PredictResponse(BaseModel):
    predictions: dict[str, int]   # e.g. {"toxic": 1, "obscene": 0, ...}
    text_received: str            # echo back so client can verify