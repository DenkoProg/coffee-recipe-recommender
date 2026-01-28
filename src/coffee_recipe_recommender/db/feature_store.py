"""SQLite-based feature store for pre-computed features."""

import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from coffee_recipe_recommender.preprocessing.preprocessing import FeatureEngineer


class FeatureStore:
    """
    SQLite-based feature store for pre-computed statistics and pair features.

    Stores:
    - user_stats, recipe_stats, temporal_behavioral_stats, global_stats (for training)
    - pair_features: pre-computed features for ALL user-recipe pairs (for fast inference)
    """

    TABLES = ["user_stats", "recipe_stats", "user_temporal_behavioral_stats", "global_stats"]

    def __init__(self, db_path: str | Path):
        """
        Initialize feature store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def exists(self) -> bool:
        """Check if feature store database exists and has all required tables."""
        if not self.db_path.exists():
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            conn.close()
            return all(table in existing_tables for table in self.TABLES)
        except sqlite3.Error:
            return False

    def has_pair_features(self) -> bool:
        """Check if pre-computed pair features table exists."""
        if not self.db_path.exists():
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pair_features'")
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except sqlite3.Error:
            return False

    def save(self, feature_engineer: "FeatureEngineer") -> None:
        """
        Save all computed stats from a fitted FeatureEngineer.

        Args:
            feature_engineer: FeatureEngineer instance with computed stats
        """
        if feature_engineer.user_stats is None:
            raise ValueError("FeatureEngineer has no user_stats. Call fit() first.")
        if feature_engineer.recipe_stats is None:
            raise ValueError("FeatureEngineer has no recipe_stats. Call fit() first.")
        if feature_engineer.user_temporal_behavioral_stats is None:
            raise ValueError("FeatureEngineer has no user_temporal_behavioral_stats. Call fit() first.")
        if feature_engineer.global_stats is None:
            raise ValueError("FeatureEngineer has no global_stats. Call fit() first.")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)

        try:
            feature_engineer.user_stats.to_sql("user_stats", conn, if_exists="replace", index=False)
            feature_engineer.recipe_stats.to_sql("recipe_stats", conn, if_exists="replace", index=False)
            feature_engineer.user_temporal_behavioral_stats.to_sql(
                "user_temporal_behavioral_stats", conn, if_exists="replace", index=False
            )
            # Save global_stats as a single-row table with JSON
            global_stats_df = pd.DataFrame([{"data": json.dumps(feature_engineer.global_stats)}])
            global_stats_df.to_sql("global_stats", conn, if_exists="replace", index=False)
            print(f"✅ Feature store saved to {self.db_path}")
        finally:
            conn.close()

    def save_pair_features(self, pair_features_df: pd.DataFrame) -> None:
        """
        Save pre-computed pair features for all user-recipe combinations.

        Args:
            pair_features_df: DataFrame with user_id, recipe_id, and all feature columns
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)

        try:
            # Create table with index on (user_id, recipe_id) for fast lookups
            pair_features_df.to_sql("pair_features", conn, if_exists="replace", index=False)
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pair_user_recipe ON pair_features(user_id, recipe_id)")
            conn.commit()
            print(f"✅ Pair features saved: {len(pair_features_df)} rows")
        finally:
            conn.close()

    def load_stats(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Load all stats from SQLite.

        Returns:
            Tuple of (user_stats, recipe_stats, user_temporal_behavioral_stats, global_stats)
        """
        if not self.exists():
            raise FileNotFoundError(f"Feature store not found at {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            user_stats = pd.read_sql("SELECT * FROM user_stats", conn)
            recipe_stats = pd.read_sql("SELECT * FROM recipe_stats", conn)
            user_temporal_behavioral_stats = pd.read_sql("SELECT * FROM user_temporal_behavioral_stats", conn)
            global_stats_df = pd.read_sql("SELECT * FROM global_stats", conn)
            global_stats = json.loads(global_stats_df.iloc[0]["data"])
            print(f"✅ Feature store loaded from {self.db_path}")
            return user_stats, recipe_stats, user_temporal_behavioral_stats, global_stats
        finally:
            conn.close()

    def get_pair_features(self, user_id: str, recipe_ids: list[str]) -> pd.DataFrame:
        """
        Get pre-computed features for a user and list of recipe candidates.

        Args:
            user_id: User identifier
            recipe_ids: List of recipe IDs to get features for

        Returns:
            DataFrame with features for each (user_id, recipe_id) pair
        """
        conn = self._get_connection()
        placeholders = ",".join(["?"] * len(recipe_ids))
        query = f"SELECT * FROM pair_features WHERE user_id = ? AND recipe_id IN ({placeholders})"
        params = [str(user_id)] + [str(rid) for rid in recipe_ids]
        return pd.read_sql(query, conn, params=params)

    def get_all_pair_features_for_user(self, user_id: str) -> pd.DataFrame:
        """
        Get all pre-computed features for a single user.

        Args:
            user_id: User identifier

        Returns:
            DataFrame with features for all recipes for this user
        """
        conn = self._get_connection()
        query = "SELECT * FROM pair_features WHERE user_id = ?"
        return pd.read_sql(query, conn, params=[str(user_id)])
