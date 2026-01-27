from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path
import json
import re
import random
import csv

class RecipeOut(BaseModel):
    recipe_id: str
    score: Optional[float] = None
    name: Optional[str] = None
    description: Optional[str] = None
    taste_bitterness: Optional[float] = None
    taste_sweetness: Optional[float] = None
    taste_acidity: Optional[float] = None
    taste_body: Optional[float] = None
    strength: Optional[str] = None
    portion_size_ml: Optional[str] = None
    preparation_time_minutes: Optional[str] = None
    difficulty: Optional[str] = None
    required_equipment: List[str] = []
    tags: List[str] = []


class RecommendOut(BaseModel):
    user_id: str
    recommendations: List[RecipeOut]
    took_ms: float


def recommend(user_id: str, n: int) -> List[Tuple[str, float]]:
    RECIPES_CSV_PATH = Path("data/recipes.csv")
    if n <= 0:
        return []

    if not RECIPES_CSV_PATH.exists():
        raise FileNotFoundError(f"{RECIPES_CSV_PATH} not found")

    recipe_ids: List[str] = []

    with RECIPES_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "recipe_id" not in reader.fieldnames:
            raise ValueError("recipes.csv must contain a 'recipe_id' column")

        for row in reader:
            rid = (row.get("recipe_id") or "").strip()
            if rid:
                recipe_ids.append(rid)

    if not recipe_ids:
        return []

    k = min(n, len(recipe_ids))
    sampled = random.sample(recipe_ids, k=k)

    # score is always 0.0 for now
    return [(rid, 0.0) for rid in sampled]

def get_info(recs: List[Tuple[str, float]] | List[dict]) -> List[dict]:
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
    recipes: Dict[str, Dict[str, str]] = {}
    with RECIPES_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "recipe_id" not in reader.fieldnames:
            raise ValueError("recipes.csv must contain a 'recipe_id' column")
        for row in reader:
            rid = (row.get("recipe_id") or "").strip()
            if not rid:
                continue
            recipes[rid] = row

    def parse_list_field(value: Any) -> List[str]:
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

    out: List[dict] = []
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

        entry: Dict[str, Any] = {"recipe_id": recipe_id, "score": score}

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
        entry["preparation_time_minutes"] = (row.get("preparation_time_minutes") or row.get("prep_time_minutes") or "").strip() or None
        entry["difficulty"] = (row.get("difficulty") or "").strip() or None

        # Lists: equipment and tags
        entry["required_equipment"] = parse_list_field(row.get("required_equipment") or row.get("equipment"))
        entry["tags"] = parse_list_field(row.get("tags") or row.get("categories") or row.get("labels"))

        out.append(entry)

    return out