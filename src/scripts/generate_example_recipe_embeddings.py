import argparse
import csv
import json
from pathlib import Path

import numpy as np


MAX_PORTION_ML = 500.0
MAX_PREP_MINUTES = 1440.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simple recipe embeddings from recipes.csv for pipeline testing."
    )
    parser.add_argument(
        "--recipes-csv",
        type=Path,
        default=Path("data/recipes.csv"),
        help="Path to recipes.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/recipe_embeddings.npy"),
        help="Output .npy embeddings file.",
    )
    return parser.parse_args()


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def difficulty_onehot(value: str | None) -> list[float]:
    if not value:
        return [0.0, 0.0, 0.0]
    normalized = value.strip().lower()
    if normalized == "beginner":
        return [1.0, 0.0, 0.0]
    if normalized == "intermediate":
        return [0.0, 1.0, 0.0]
    if normalized == "advanced":
        return [0.0, 0.0, 1.0]
    return [0.0, 0.0, 0.0]


def parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def build_embedding(row: dict[str, str], equipment_vocab: list[str]) -> list[float]:
    taste_bitterness = to_float(row.get("taste_bitterness"))
    taste_sweetness = to_float(row.get("taste_sweetness"))
    taste_acidity = to_float(row.get("taste_acidity"))
    taste_body = to_float(row.get("taste_body"))

    strength = to_float(row.get("strength"))
    strength_norm = strength / 5.0 if strength else 0.0

    portion_size = to_float(row.get("portion_size_ml"))
    portion_norm = min(portion_size, MAX_PORTION_ML) / MAX_PORTION_ML if portion_size else 0.0

    prep_minutes = to_float(row.get("preparation_time_minutes"))
    prep_norm = min(prep_minutes, MAX_PREP_MINUTES) / MAX_PREP_MINUTES if prep_minutes else 0.0

    difficulty = difficulty_onehot(row.get("difficulty"))

    required_equipment = set(parse_json_list(row.get("required_equipment")))
    equipment_vector = [1.0 if item in required_equipment else 0.0 for item in equipment_vocab]

    return [
        taste_bitterness,
        taste_sweetness,
        taste_acidity,
        taste_body,
        strength_norm,
        portion_norm,
        prep_norm,
        *difficulty,
        *equipment_vector,
    ]


def main() -> None:
    args = parse_args()
    recipes_path = args.recipes_csv.expanduser()
    output_path = args.output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    equipment_set: set[str] = set()

    with recipes_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "recipe_id" not in reader.fieldnames:
            raise ValueError("recipes.csv must include a recipe_id column.")
        for row in reader:
            rows.append(row)
            equipment_set.update(parse_json_list(row.get("required_equipment")))

    if not rows:
        raise ValueError("No recipes found in CSV; cannot generate embeddings.")

    equipment_vocab = sorted(equipment_set)

    embeddings = [build_embedding(row, equipment_vocab) for row in rows]
    array = np.array(embeddings, dtype=np.float32)
    np.save(output_path, array)

    print(
        f"Saved {array.shape[0]} embeddings with dim {array.shape[1]} to {output_path}. "
        f"Equipment vocab size: {len(equipment_vocab)}."
    )


if __name__ == "__main__":
    main()
