from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'purchase_intention_model.joblib'
FEATURE_COLUMNS = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'SpecialDay',
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend',
]

app = FastAPI(title='E-commerce Purchase Intention Prediction API')


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model artifact not found at {MODEL_PATH}. Run train.py first.')
    return joblib.load(MODEL_PATH)


app.state.model = load_model()


class PurchaseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0.0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0.0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0.0)
    BounceRates: float = Field(ge=0.0, le=1.0)
    ExitRates: float = Field(ge=0.0, le=1.0)
    SpecialDay: float = Field(ge=0.0, le=1.0)
    Month: str
    OperatingSystems: int = Field(ge=1)
    Browser: int = Field(ge=1)
    Region: int = Field(ge=1)
    TrafficType: int = Field(ge=1)
    VisitorType: str
    Weekend: bool

    @property
    def as_dataframe(self) -> pd.DataFrame:
        payload = self.model_dump()
        return pd.DataFrame([payload], columns=FEATURE_COLUMNS)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'model': 'purchase_intention_model'}


@app.post('/predict')
def predict(request: PurchaseRequest) -> dict[str, object]:
    model = app.state.model
    frame = request.as_dataframe
    prediction = model.predict(frame)[0]
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        'prediction': bool(prediction),
        'predicted_class': bool(prediction),
        'probability': probability,
    }


if __name__ == '__main__':
    uvicorn.run('predict:app', host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
