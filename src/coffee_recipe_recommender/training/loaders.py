import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_json_column(series: pd.Series) -> pd.Series:
    """Parse JSON string column to Python objects."""

    def safe_parse(value: Any) -> Any:
        if pd.isna(value) or value == "":
            return None
        if isinstance(value, (list | dict)):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    return series.apply(safe_parse)


def load_users(path: Path) -> pd.DataFrame:
    """
    Load users.csv with proper dtypes and JSON parsing.

    Returns:
        DataFrame with columns: user_id, username, owned_equipment (list),
        available_products (list), taste_pref_* (float), preferred_strength (int),
        preferred_portion_size (str), dietary_restrictions (list), account_created (datetime)
    """
    path = path.expanduser()
    df = pd.read_csv(path)

    json_columns = ["owned_equipment", "available_products", "dietary_restrictions"]
    for col in json_columns:
        if col in df.columns:
            df[col] = parse_json_column(df[col])

    if "account_created" in df.columns:
        df["account_created"] = pd.to_datetime(df["account_created"])

    taste_cols = [
        "taste_pref_bitterness",
        "taste_pref_sweetness",
        "taste_pref_acidity",
        "taste_pref_body",
    ]
    for col in taste_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def load_recipes(path: Path) -> pd.DataFrame:
    """
    Load recipes.csv with proper dtypes and JSON parsing.

    Returns:
        DataFrame with columns: recipe_id, name, description,
        taste_* (float), strength (int), portion_size_ml (int),
        preparation_time_minutes (int), difficulty (str),
        required_equipment (list), required_products (dict), tags (list)
    """
    path = path.expanduser()
    df = pd.read_csv(path)

    json_columns = ["required_equipment", "required_products", "tags"]
    for col in json_columns:
        if col in df.columns:
            df[col] = parse_json_column(df[col])
    taste_cols = ["taste_bitterness", "taste_sweetness", "taste_acidity", "taste_body"]
    for col in taste_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    int_cols = ["strength", "portion_size_ml", "preparation_time_minutes"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def load_interactions(path: Path) -> pd.DataFrame:
    """
    Load interactions CSV with proper dtypes.

    Returns:
        DataFrame with columns: interaction_id, user_id, recipe_id,
        timestamp (datetime), rating (float, nullable), completed (bool)
    """
    path = path.expanduser()
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "completed" in df.columns:
        df["completed"] = df["completed"].astype(bool)

    return df


def create_id_mappings(
    users_df: pd.DataFrame, recipes_df: pd.DataFrame
) -> tuple[dict[str, int], dict[str, int], dict[int, str], dict[int, str]]:
    """
    Create bidirectional mappings from string IDs to integer indices.

    Args:
        users_df: Users DataFrame with user_id column
        recipes_df: Recipes DataFrame with recipe_id column

    Returns:
        Tuple of (user_to_idx, recipe_to_idx, idx_to_user, idx_to_recipe)
    """
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(users_df["user_id"].unique()))}
    recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(sorted(recipes_df["recipe_id"].unique()))}

    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_recipe = {idx: recipe_id for recipe_id, idx in recipe_to_idx.items()}

    return user_to_idx, recipe_to_idx, idx_to_user, idx_to_recipe
