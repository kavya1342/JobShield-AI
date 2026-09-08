from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "jobshield_model.joblib"


def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. "
            "Run 'python -m src.train' first."
        )

    return joblib.load(MODEL_PATH)


MODEL_BUNDLE = load_model_bundle()
MODEL = MODEL_BUNDLE["model"]
THRESHOLD = MODEL_BUNDLE["threshold"]
MODEL_VERSION = MODEL_BUNDLE["version"]


def predict_posting(posting: dict[str, Any]) -> dict[str, Any]:
    posting_frame = pd.DataFrame([posting])

    fraud_score = float(
        MODEL.predict_proba(posting_frame)[0, 1]
    )

    predicted_class = int(
        fraud_score >= THRESHOLD
    )

    return {
        "fraud_score": round(fraud_score, 4),
        "predicted_class": predicted_class,
        "prediction": (
            "potentially_fraudulent"
            if predicted_class == 1
            else "likely_legitimate"
        ),
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
    }


if __name__ == "__main__":
    sample_posting = {
        "title": "Work From Home Data Entry",
        "company_profile": "",
        "description": (
            "Earn money immediately. Pay a registration "
            "fee to receive your appointment letter."
        ),
        "requirements": "No experience required",
        "benefits": "Guaranteed weekly income",
    }

    result = predict_posting(sample_posting)
    print(result)