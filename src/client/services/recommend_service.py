import csv
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class RecipeOut(BaseModel):
    recipe_id: str
    score: float | None = None
    name: str | None = None
    description: str | None = None
    taste_bitterness: float | None = None
    taste_sweetness: float | None = None
    taste_acidity: float | None = None
    taste_body: float | None = None
    strength: str | None = None
    portion_size_ml: str | None = None
    preparation_time_minutes: str | None = None
    difficulty: str | None = None
    required_equipment: list[str] = []
    tags: list[str] = []
    why_recommended: dict | None = None
    missing_equipment: list[str] = []
    user_equipment_matched: bool | None = None


class RecommendOut(BaseModel):
    user_id: str
    recommendations: list[RecipeOut]
    took_ms: float


def recommend(user_id: str, n: int = 5) -> list[tuple[str, float]]:
    """Generate top-N recipe recommendations for a user.

    Args:
        user_id: Target user identifier
        n: Number of recommendations to return

    Returns:
        List of (recipe_id, score) tuples, sorted by score descending.
    """
    import pandas as pd

    from coffee_recipe_recommender.inference.recommender import Recommender

    # Load dataframes
    users_df = pd.read_csv("data/users.csv")
    recipes_df = pd.read_csv("data/recipes.csv")
    train_df = pd.read_csv("data/interactions_train_split.csv")

    # Load model (hybrid by default)
    recommender = Recommender.from_hybrid_checkpoints(
        retrieval_checkpoint_path=Path("runs/retrieval/baseline_with_features/retrieval_final.pt"),
        ranker_model_path=Path("runs/ranking/improved-features/ranker.pkl"),
        vector_store_path=Path("data/chroma"),
        feature_store_path=Path("data/feature_store.db"),
        users_df=users_df,
        recipes_df=recipes_df,
        cold_start_path=Path("runs/retrieval/cold_encoder_baseline/cold_encoder.pt"),
    )

    # Generate recommendations
    recommendations = recommender.recommend(
        user_id=user_id,
        users_df=users_df,
        recipes_df=recipes_df,
        train_df=train_df,
        n=n,
    )

    return recommendations[:n] if recommendations else []


def get_info(
    user_id: str,
    recs: list[tuple[str, float]] | list[dict],
    recipes_csv_path: Path = Path("data/recipes.csv"),
    users_csv_path: Path = Path("data/users.csv"),
) -> list[dict]:
    """
    Enrich recommendations with recipe metadata AND equipment-compat info for the given user.

    Adds:
      - missing_equipment: list[str]
      - user_equipment_matched: bool

    users.csv columns (relevant):
      - user_id
      - owned_equipment (JSON array)
    """
    if not recipes_csv_path.exists():
        raise FileNotFoundError(f"{recipes_csv_path} not found")
    if not users_csv_path.exists():
        raise FileNotFoundError(f"{users_csv_path} not found")

    def parse_list_field(value: Any) -> list[str]:
        if value is None:
            return []
        s = str(value).strip()
        if not s:
            return []
        # JSON array (your owned_equipment format)
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except Exception:
                pass
        # fallback (if some rows are not JSON for any reason)
        return [p.strip() for p in re.split(r"[|,;]", s) if p.strip()]

    def try_float(v: Any):
        try:
            if v is None or str(v).strip() == "":
                return None
            return float(v)
        except Exception:
            return None

    def norm_equip(x: str) -> str:
        # normalize for matching: "Gooseneck Kettle" == "gooseneck_kettle"
        s = str(x).strip().lower()
        s = re.sub(r"[\s\-]+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s

    # ---- load user owned_equipment ----
    user_row: dict[str, str] | None = None
    with users_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "user_id" not in reader.fieldnames:
            raise ValueError("users.csv must contain a 'user_id' column")
        for row in reader:
            if (row.get("user_id") or "").strip() == user_id:
                user_row = row
                break

    if user_row is None:
        raise ValueError(f"User '{user_id}' not found in {users_csv_path}")

    owned_equipment = parse_list_field(user_row.get("owned_equipment"))
    owned_norm = {norm_equip(x) for x in owned_equipment}

    # ---- load recipes lookup ----
    recipes: dict[str, dict[str, str]] = {}
    with recipes_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "recipe_id" not in reader.fieldnames:
            raise ValueError("recipes.csv must contain a 'recipe_id' column")
        for row in reader:
            rid = (row.get("recipe_id") or "").strip()
            if rid:
                recipes[rid] = row

    # ---- enrich ----
    out: list[dict] = []
    for r in recs:
        if isinstance(r, (list, tuple)):
            recipe_id = str(r[0]) if len(r) > 0 else ""
            score = r[1] if len(r) > 1 else None
        elif isinstance(r, dict):
            recipe_id = str(r.get("recipe_id") or "")
            score = r.get("score")
        else:
            continue

        entry: dict[str, Any] = {"recipe_id": recipe_id, "score": score}

        row = recipes.get(recipe_id)
        if not row:
            entry["required_equipment"] = []
            entry["missing_equipment"] = []
            entry["user_equipment_matched"] = True
            out.append(entry)
            continue

        entry["name"] = (row.get("name") or "").strip() or None
        entry["description"] = (row.get("description") or "").strip() or None

        entry["taste_bitterness"] = try_float(row.get("taste_bitterness") or row.get("bitterness"))
        entry["taste_sweetness"] = try_float(row.get("taste_sweetness") or row.get("sweetness"))
        entry["taste_acidity"] = try_float(row.get("taste_acidity") or row.get("acidity"))
        entry["taste_body"] = try_float(row.get("taste_body") or row.get("body"))

        entry["strength"] = (row.get("strength") or "").strip() or None
        entry["portion_size_ml"] = (row.get("portion_size_ml") or row.get("portion") or "").strip() or None
        entry["preparation_time_minutes"] = (
            row.get("preparation_time_minutes") or row.get("prep_time_minutes") or ""
        ).strip() or None
        entry["difficulty"] = (row.get("difficulty") or "").strip() or None

        required = parse_list_field(row.get("required_equipment") or row.get("equipment"))
        entry["required_equipment"] = required
        entry["tags"] = parse_list_field(row.get("tags") or row.get("categories") or row.get("labels"))

        missing = [req for req in required if norm_equip(req) not in owned_norm]
        entry["missing_equipment"] = missing
        entry["user_equipment_matched"] = (len(missing) == 0)

        out.append(entry)

    return out
