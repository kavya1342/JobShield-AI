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


def predict_posting(
    posting: dict[str, Any]
) -> dict[str, Any]:
    posting_frame = pd.DataFrame([posting])

    fraud_score = float(
        MODEL.predict_proba(posting_frame)[0, 1]
    )

    combined_text = MODEL.named_steps[
        "combine_text"
    ].transform(posting_frame)

    vectorizer = MODEL.named_steps["tfidf"]
    classifier = MODEL.named_steps["classifier"]

    posting_vector = vectorizer.transform(
        combined_text
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )
    weights = classifier.coef_[0]

    contributions = (
        posting_vector
        .multiply(weights)
        .toarray()[0]
    )

    positive_indices = [
        index
        for index, contribution
        in enumerate(contributions)
        if contribution > 0
    ]

    top_indices = sorted(
        positive_indices,
        key=lambda index: contributions[index],
        reverse=True
    )[:5]

    top_risk_terms = [
        {
            "term": str(feature_names[index]),
            "contribution": round(
                float(contributions[index]),
                4
            ),
        }
        for index in top_indices
    ]

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
        "top_risk_terms": top_risk_terms,
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