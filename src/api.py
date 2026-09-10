import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        (
            "http://localhost:5173,"
            "http://127.0.0.1:5173"
        ),
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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