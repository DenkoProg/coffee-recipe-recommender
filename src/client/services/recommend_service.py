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


def get_info(recs: list[tuple[str, float]] | list[dict]) -> list[dict]:
    """
    Enrich a list of recommendations with recipe metadata from `data/recipes.csv`.

    Accepts input in either of these forms:
      - List of tuples: [(recipe_id, score), ...]
      - List of dicts: [{'recipe_id': ..., 'score': ...}, ...]

    Returns a list of dicts containing at least: recipe_id, score, name, description,
    taste_bitterness, taste_sweetness, taste_acidity, taste_body, strength,
    portion_size_ml, preparation_time_minutes, difficulty, required_equipment (list), tags (list).
    """
    RECIPES_CSV_PATH = Path("data/recipes.csv")
    if not RECIPES_CSV_PATH.exists():
        raise FileNotFoundError(f"{RECIPES_CSV_PATH} not found")

    # Load recipes into a lookup by recipe_id
    recipes: dict[str, dict[str, str]] = {}
    with RECIPES_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "recipe_id" not in reader.fieldnames:
            raise ValueError("recipes.csv must contain a 'recipe_id' column")
        for row in reader:
            rid = (row.get("recipe_id") or "").strip()
            if not rid:
                continue
            recipes[rid] = row

    def parse_list_field(value: Any) -> list[str]:
        if value is None:
            return []
        s = str(value).strip()
        if not s:
            return []
        # try JSON list first
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                pass
        # split by common delimiters
        parts = [p.strip() for p in re.split(r"[|,;]", s) if p.strip()]
        return parts

    def try_float(v: Any):
        try:
            if v is None or str(v).strip() == "":
                return None
            return float(v)
        except Exception:
            return None

    out: list[dict] = []
    for r in recs:
        if isinstance(r, (list, tuple)):
            recipe_id = str(r[0]) if len(r) > 0 else ""
            score = r[1] if len(r) > 1 else None
        elif isinstance(r, dict):
            recipe_id = str(r.get("recipe_id") or "")
            score = r.get("score")
        else:
            # unsupported entry, skip
            continue

        entry: dict[str, Any] = {"recipe_id": recipe_id, "score": score}

        row = recipes.get(recipe_id)
        if not row:
            out.append(entry)
            continue

        # Basic textual fields
        entry["name"] = (row.get("name") or "").strip() or None
        entry["description"] = (row.get("description") or "").strip() or None

        # Taste fields (attempt to parse floats)
        entry["taste_bitterness"] = try_float(row.get("taste_bitterness") or row.get("bitterness"))
        entry["taste_sweetness"] = try_float(row.get("taste_sweetness") or row.get("sweetness"))
        entry["taste_acidity"] = try_float(row.get("taste_acidity") or row.get("acidity"))
        entry["taste_body"] = try_float(row.get("taste_body") or row.get("body"))

        # Other presentational fields
        entry["strength"] = (row.get("strength") or "").strip() or None
        entry["portion_size_ml"] = (row.get("portion_size_ml") or row.get("portion") or "").strip() or None
        entry["preparation_time_minutes"] = (
            row.get("preparation_time_minutes") or row.get("prep_time_minutes") or ""
        ).strip() or None
        entry["difficulty"] = (row.get("difficulty") or "").strip() or None

        # Lists: equipment and tags
        entry["required_equipment"] = parse_list_field(row.get("required_equipment") or row.get("equipment"))
        entry["tags"] = parse_list_field(row.get("tags") or row.get("categories") or row.get("labels"))

        out.append(entry)

    return out
