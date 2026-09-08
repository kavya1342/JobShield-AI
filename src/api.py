from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import (
    MODEL_VERSION,
    THRESHOLD,
    predict_posting,
)


app = FastAPI(
    title="JobShield AI API",
    description=(
        "Detect potentially fraudulent job postings "
        "using TF-IDF and logistic regression."
    ),
    version=MODEL_VERSION,
)


class JobPostingRequest(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=500
    )
    company_profile: str = Field(
        default="",
        max_length=20_000
    )
    description: str = Field(
        min_length=1,
        max_length=50_000
    )
    requirements: str = Field(
        default="",
        max_length=30_000
    )
    benefits: str = Field(
        default="",
        max_length=30_000
    )

class RiskTerm(BaseModel):
    term: str
    contribution: float

class PredictionResponse(BaseModel):
    fraud_score: float
    predicted_class: int
    prediction: str
    threshold: float
    model_version: str
    top_risk_terms: list[RiskTerm]


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    posting: JobPostingRequest
) -> PredictionResponse:
    result = predict_posting(
        posting.model_dump()
    )

    return PredictionResponse(**result)