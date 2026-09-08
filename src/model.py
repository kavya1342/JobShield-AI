from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import (
    CLASS_WEIGHT,
    LOGISTIC_REGRESSION_C,
    MAX_ITERATIONS,
    MAX_TFIDF_FEATURES,
    NGRAM_RANGE,
    RANDOM_STATE,
)
from .text_processing import combine_text_columns

CUSTOM_STOP_WORDS = sorted(
    ENGLISH_STOP_WORDS
    - {"no", "not", "nor", "never"}
)
def build_model() -> Pipeline:
    """Create an unfitted JobShield model pipeline."""

    return Pipeline([
        (
            "combine_text",
            FunctionTransformer(
                combine_text_columns,
                validate=False
            )
        ),
        (
            "tfidf",
            TfidfVectorizer(
                max_features=MAX_TFIDF_FEATURES,
                ngram_range=NGRAM_RANGE,
                stop_words=CUSTOM_STOP_WORDS
            )
        ),
        (
            "classifier",
            LogisticRegression(
                C=LOGISTIC_REGRESSION_C,
                class_weight=CLASS_WEIGHT,
                max_iter=MAX_ITERATIONS,
                random_state=RANDOM_STATE
            )
        )
    ])