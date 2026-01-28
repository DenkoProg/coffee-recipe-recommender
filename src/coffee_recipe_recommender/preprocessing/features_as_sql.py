import sqlite3
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from coffee_recipe_recommender.preprocessing.preprocessing import FeatureEngineer


def save_to_sqlite(
    fe: "FeatureEngineer",
    train_features: pd.DataFrame,
    db_path: str = "coffee_features.db",
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        fe.user_stats.to_sql("user_stats", conn, if_exists="replace", index=False)
        fe.recipe_stats.to_sql("recipe_stats", conn, if_exists="replace", index=False)
        fe.user_temporal_behavioral_stats.to_sql(
            "user_temporal_behavioral_stats", conn, if_exists="replace", index=False
        )
        train_features.to_sql("train_features", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def save_to_postgresql(
    fe: "FeatureEngineer",
    train_features: pd.DataFrame,
    connection_string: str,
) -> None:
    from sqlalchemy import create_engine

    engine: "Engine" = create_engine(connection_string)

    fe.user_stats.to_sql("user_stats", engine, if_exists="replace", index=False)
    fe.recipe_stats.to_sql("recipe_stats", engine, if_exists="replace", index=False)
    fe.user_temporal_behavioral_stats.to_sql(
        "user_temporal_behavioral_stats", engine, if_exists="replace", index=False
    )
    train_features.to_sql("train_features", engine, if_exists="replace", index=False)
