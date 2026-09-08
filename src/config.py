TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

TARGET_COLUMN = "fraudulent"

MAX_TFIDF_FEATURES = 5000
NGRAM_RANGE = (1, 1)

LOGISTIC_REGRESSION_C = 2.0
CLASS_WEIGHT = "balanced"
MAX_ITERATIONS = 1000

DECISION_THRESHOLD = 0.60
RANDOM_STATE = 42
MODEL_VERSION = "0.2.0"