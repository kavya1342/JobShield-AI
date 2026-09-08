from pathlib import Path

import joblib
import pandas as pd

from .config import (
    DECISION_THRESHOLD,
    MODEL_VERSION,
    TARGET_COLUMN,
)
from .model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "jobshield_model.joblib"


def main() -> None:
    train_df = pd.read_csv(
        PROCESSED_DATA_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        PROCESSED_DATA_DIR / "validation.csv"
    )

    development_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True
    )

    x_development = development_df.drop(
        columns=[TARGET_COLUMN]
    )
    y_development = development_df[TARGET_COLUMN]

    model = build_model()
    model.fit(x_development, y_development)

    model_bundle = {
        "model": model,
        "threshold": DECISION_THRESHOLD,
        "version": MODEL_VERSION,
        "training_rows": len(development_df),
        "training_sources": [
            "train.csv",
            "validation.csv",
        ],
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model_bundle, MODEL_PATH)

    print("Production training completed")
    print(f"Training rows: {len(development_df)}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()