"""Build SQLite feature store from training data."""

import argparse
from pathlib import Path

import pandas as pd

from coffee_recipe_recommender.db.feature_store import FeatureStore
from coffee_recipe_recommender.preprocessing.features import FeatureEngineer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite feature store from training data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/feature_store.db"),
        help="Output path for SQLite feature store",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("📂 Loading data...")
    users = pd.read_csv(args.data_dir / "users.csv")
    recipes = pd.read_csv(args.data_dir / "recipes.csv")

    # Use split train data if available, otherwise use full train
    train_path = args.data_dir / "interactions_train_split.csv"
    if not train_path.exists():
        train_path = args.data_dir / "interactions_train.csv"
    train_interactions = pd.read_csv(train_path)

    print(f"📊 Users: {len(users)}, Recipes: {len(recipes)}, Interactions: {len(train_interactions)}")

    print("🔧 Fitting FeatureEngineer...")
    fe = FeatureEngineer()
    fe.fit(users, recipes, train_interactions)

    print("💾 Saving to feature store...")
    store = FeatureStore(args.output)
    store.save(fe)

    print(f"✅ Feature store built at {args.output}")
    print(f"   - user_stats: {len(fe.user_stats)} rows")
    print(f"   - recipe_stats: {len(fe.recipe_stats)} rows")
    print(f"   - user_temporal_behavioral_stats: {len(fe.user_temporal_behavioral_stats)} rows")


if __name__ == "__main__":
    main()
