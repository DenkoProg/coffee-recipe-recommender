import argparse
from pathlib import Path

import pandas as pd
import torch

from coffee_recipe_recommender.evaluation.metrics import evaluate_recommendations
from coffee_recipe_recommender.inference.recommender import Recommender
from coffee_recipe_recommender.training.loaders import load_interactions, load_recipes, load_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval model performance.")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--cold-start-path", type=Path, help="Path to cold-start encoder checkpoint")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to recipe embeddings .npy file")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CSV files")
    parser.add_argument(
        "--mode", type=str, choices=["retrieval", "hybrid"], default="retrieval", help="Recommender mode"
    )
    parser.add_argument("--ranker-model", type=Path, help="Path to ranker model (required for hybrid mode)")
    parser.add_argument(
        "--eval-split", type=str, choices=["val", "val_cold"], default="val", help="Evaluation split to use"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations (for metrics @k)")
    parser.add_argument("--min-rating", type=float, default=3.5, help="Minimum rating for positive interaction")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference"
    )
    parser.add_argument("--output-file", type=Path, help="Optional: Save metrics to JSON file")

    return parser.parse_args()


def build_ground_truth(
    interactions_df: pd.DataFrame,
    min_rating: float = 3.5,
) -> dict[str, set[str]]:
    """
    Build ground truth dictionary from interactions.

    Args:
        interactions_df: DataFrame with user_id, recipe_id, rating, completed columns
        min_rating: Minimum rating to consider as positive

    Returns:
        Dictionary mapping user_id to set of relevant recipe_ids
    """
    ground_truth: dict[str, set[str]] = {}

    for _, row in interactions_df.iterrows():
        user_id = row["user_id"]
        recipe_id = row["recipe_id"]
        rating = row.get("rating")
        completed = row.get("completed", False)

        # Consider positive if rating >= min_rating OR completed=True
        is_positive = (pd.notna(rating) and rating >= min_rating) or completed

        if is_positive:
            if user_id not in ground_truth:
                ground_truth[user_id] = set()
            ground_truth[user_id].add(recipe_id)

    return ground_truth


def main() -> None:
    args = parse_args()

    print("Loading data...")
    users_df = load_users(args.data_dir / "users.csv")
    recipes_df = load_recipes(args.data_dir / "recipes.csv")

    # Load evaluation split
    eval_file = f"interactions_{args.eval_split}.csv"
    eval_df = load_interactions(args.data_dir / eval_file)

    # Load training split for train_df parameter
    train_df = load_interactions(args.data_dir / "interactions_train.csv")

    print(f"\nEvaluation split: {args.eval_split}")
    print(f"  Users: {eval_df['user_id'].nunique()}")
    print(f"  Interactions: {len(eval_df)}")

    # Build ground truth
    print("\nBuilding ground truth...")
    ground_truth = build_ground_truth(eval_df, args.min_rating)
    print(f"  Users with positive interactions: {len(ground_truth)}")

    # Load recommender
    print(f"\nLoading {args.mode} recommender from {args.checkpoint}...")
    if args.cold_start_path:
        print(f"  with cold-start encoder from {args.cold_start_path}...")
    if args.mode == "retrieval":
        recommender = Recommender.from_retrieval_checkpoint(
            checkpoint_path=args.checkpoint,
            cold_start_path=args.cold_start_path,
            embeddings_path=args.embeddings,
            users_df=users_df,
            device=args.device,
        )
    else:  # hybrid
        if not args.ranker_model:
            raise ValueError("--ranker-model is required for hybrid mode")
        recommender = Recommender.from_hybrid_checkpoints(
            retrieval_checkpoint_path=args.checkpoint,
            ranker_model_path=args.ranker_model,
            cold_start_path=args.cold_start_path,
            embeddings_path=args.embeddings,
            users_df=users_df,
            recipes_df=recipes_df,
            device=args.device,
        )

    # Generate recommendations
    print(f"\nGenerating top-{args.k} recommendations for each user...")
    user_ids = list(ground_truth.keys())

    recommendations_dict = {}
    for user_id in user_ids:
        try:
            recs = recommender.recommend(user_id, users_df, recipes_df, train_df, n=args.k)
            recommendations_dict[user_id] = [recipe_id for recipe_id, _ in recs]
        except ValueError:
            # Unknown user (cold-start case not in training)
            continue

    print(f"  Generated recommendations for {len(recommendations_dict)} users")

    # Evaluate
    print("\nEvaluating recommendations...")
    metrics = evaluate_recommendations(
        recommendations_dict=recommendations_dict,
        ground_truth_dict=ground_truth,
        k=args.k,
        catalog_size=len(recipes_df),
    )

    # Print results
    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 50}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<20}: {value:.4f}")
    print(f"{'=' * 50}\n")

    # Check if we hit the target
    ndcg_key = f"NDCG@{args.k}"
    target_ndcg = 0.4
    if ndcg_key in metrics:
        if metrics[ndcg_key] >= target_ndcg:
            print(f"✅ SUCCESS: {ndcg_key} = {metrics[ndcg_key]:.4f} (target: {target_ndcg})")
        else:
            print(f"⚠️  {ndcg_key} = {metrics[ndcg_key]:.4f} < {target_ndcg} (target not met)")

    # Save metrics if requested
    if args.output_file:
        import json

        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output_file}")


if __name__ == "__main__":
    main()
