

from pydantic import BaseModel

import pandas as pd
from typing import Optional

class HistoryItem(BaseModel):
    timestamp: str
    recipe_id: str
    rating: Optional[float] = None
    completed: bool

INTERACTIONS_PATH = "data/interactions_train.csv"   # change if needed

def get_user_history(user_id: str, limit: int = 50) -> list[dict]:
    """
    Read interactions.csv, filter by user_id, sort by timestamp desc,
    return list of dicts with keys: timestamp, recipe_id, rating, completed
    """

    df = pd.read_csv(INTERACTIONS_PATH)

    user_df = df[df["user_id"] == user_id]

    if user_df.empty:
        return []

    user_df["timestamp"] = pd.to_datetime(user_df["timestamp"], errors="coerce")
    user_df = user_df.sort_values("timestamp", ascending=False)

    user_df = user_df.head(limit)

    return [
        {
            "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "recipe_id": row["recipe_id"],
            "rating": None if pd.isna(row["rating"]) else float(row["rating"]),
            "completed": bool(row["completed"]),
        }
        for _, row in user_df.iterrows()
    ]
