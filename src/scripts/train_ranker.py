import os

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import shap

from coffee_recipe_recommender.models.ranking import LightGBMRankerModel
from coffee_recipe_recommender.preprocessing.preprocessing import FeatureEngineer, generate_training_data


def main():
    print("📂 Loading raw data...")
    users = pd.read_csv("data/users.csv")
    recipes = pd.read_csv("data/recipes.csv")
    interactions = pd.read_csv("data/interactions_train.csv")

    all_recipes = recipes["recipe_id"].unique()
    train_df = generate_training_data(interactions, all_recipes, n_candidates=50)  # Постав 50-100

    print("🛠 Generating features...")
    fe = FeatureEngineer()
    X = fe.generate(train_df, users, recipes, train_interactions_df=interactions)
    y = train_df["relevance"].astype(int)
    qids = train_df.groupby("user_id", sort=False).size().to_numpy()

    n_groups = len(qids)
    train_groups_n = int(n_groups * 0.8)

    split_idx = sum(qids[:train_groups_n])

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    qids_train, qids_val = qids[:train_groups_n], qids[train_groups_n:]

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
        model.fit(X_train, y_train, qids_train, eval_set=(X_val, y_val, qids_val))

        score = model.model.best_score_["valid_0"]["ndcg@1"]
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print(f"✅ Best params: {study.best_params}")

    print("🚀 Training final model with best params...")
    best_params = study.best_params
    best_params["objective"] = "lambdarank"
    best_params["metric"] = "ndcg"

    final_model = LightGBMRankerModel(best_params)
    final_model.fit(X, y, qids)

    os.makedirs("artifacts", exist_ok=True)
    final_model.save("artifacts/ranking_model.pkl")
    print("💾 Model saved to artifacts/ranking_model.pkl")

    print("📊 Generating SHAP values...")
    explainer = shap.TreeExplainer(final_model.model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("artifacts/shap_importance.png")
    print("🖼  SHAP plot saved to artifacts/shap_importance.png")


if __name__ == "__main__":
    main()
