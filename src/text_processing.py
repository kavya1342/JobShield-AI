import pandas as pd

from .config import TEXT_COLUMNS


def combine_text_columns(frame: pd.DataFrame) -> pd.Series:
    """
    Combine JobShield's text fields in a consistent order.

    Missing columns and missing values become empty strings.
    """
    text_frame = (
        frame
        .reindex(columns=TEXT_COLUMNS)
        .fillna("")
        .astype(str)
    )

    combined_text = (
        text_frame
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return combined_text