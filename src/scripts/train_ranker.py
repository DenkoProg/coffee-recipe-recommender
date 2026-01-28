import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import shap

from coffee_recipe_recommender.models.ranking import LightGBMRankerModel
from coffee_recipe_recommender.preprocessing.features import FeatureEngineer, generate_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker on retrieval candidates.")

    # Data paths
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CSV files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ranking/improved-features"),
        help="Directory to save trained model",
    )

    # Ranker hyperparameters
    parser.add_argument("--n-candidates", type=int, default=50, help="Number of candidates to generate per user")
    parser.add_argument("--optuna-trials", type=int, default=20, help="Number of Optuna tuning trials")

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __name__ != "__main__" else "cpu",
        help="Device (unused but for consistency)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("📂 Loading raw data...")
    users = pd.read_csv(args.data_dir / "users.csv")
    recipes = pd.read_csv(args.data_dir / "recipes.csv")
    train_interactions = pd.read_csv(args.data_dir / "interactions_train_split.csv")
    val_interactions = pd.read_csv(args.data_dir / "interactions_test_split.csv")

    all_recipes = recipes["recipe_id"].unique()
    train_df = generate_training_data(train_interactions, all_recipes, n_candidates=args.n_candidates)
    val_df = generate_training_data(val_interactions, all_recipes, n_candidates=args.n_candidates)

    print("🛠 Generating features...")
    fe = FeatureEngineer()
    fe.fit(users, recipes, train_interactions)
    X = fe.generate(train_df, users, recipes, train_interactions_df=train_interactions, verbose=True)
    X_val = fe.generate(val_df, users, recipes, train_interactions_df=train_interactions, verbose=True)

    y = train_df["relevance"].astype(int)
    y_val = val_df["relevance"].astype(int)

    qids = train_df.groupby("user_id", sort=False).size().to_numpy()
    qids_val = val_df.groupby("user_id", sort=False).size().to_numpy()

    print("🔍 Starting Optuna tuning...")

    def objective(trial):
        param = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = LightGBMRankerModel(param)
        model.fit(X, y, qids, eval_set=(X_val, y_val, qids_val))

        score = model.model.best_score_["valid_0"]["ndcg@1"]
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optuna_trials)

    print(f"✅ Best params: {study.best_params}")

    print("🚀 Training final model with best params...")
    best_params = study.best_params
    best_params["objective"] = "lambdarank"
    best_params["metric"] = "ndcg"

    final_model = LightGBMRankerModel(best_params)
    final_model.fit(X, y, qids, eval_set=(X_val, y_val, qids_val))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(str(args.output_dir / "ranker.pkl"))
    print(f"💾 Model saved to {args.output_dir / 'ranker.pkl'}")

    print("📊 Generating SHAP values...")
    explainer = shap.TreeExplainer(final_model.model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(str(args.output_dir / "shap_importance.png"))
    print(f"🖼  SHAP plot saved to {args.output_dir / 'shap_importance.png'}")


if __name__ == "__main__":
    main()
