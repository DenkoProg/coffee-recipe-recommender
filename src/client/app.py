from pathlib import Path
import time
from typing import List
import shap
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from coffee_recipe_recommender.inference.recommender import Recommender
from coffee_recipe_recommender.training.loaders import load_interactions, load_recipes, load_users
from src.client.services.recommend_service import RecommendOut, get_info, recommend
from src.client.services.users_service import UserOut, list_users

import matplotlib.pyplot as plt

app = FastAPI(title="Coffee Recommender API", version="1.0")
app.mount("/images", StaticFiles(directory="data/images"), name="images")

ART = Path("artifacts")

X_ALL = pd.read_parquet(ART / "train_features.parquet")
FEATURE_NAMES = pd.read_csv(
    ART / "feature_names.csv", header=None
)[0].tolist()

def get_feature_rows(
    user_id: str,
    recipe_ids: list[str],
    X_all: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:

    df = X_all[
        (X_all["user_id"].astype(str) == str(user_id)) &
        (X_all["recipe_id"].astype(str).isin(recipe_ids))
    ]

    if df.empty:
        raise ValueError("No feature rows found for this user and recipes")

    return df[feature_names]

from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt

def save_shap_plots_per_recommendation(
    *,
    user_id: str,
    recs: list[tuple[str, float]],
    shap_explainer: shap.TreeExplainer,
    X_all: pd.DataFrame,
    feature_names: list[str],
    out_dir: Path,
    top_k_features: int = 15,
) -> list[dict]:
    """
    Saves one SHAP plot per recommended item.

    Returns metadata list you can return from API (recipe_id + plot paths).
    """
    out_dir = out_dir / "shap" / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for rank, (recipe_id, score) in enumerate(recs, start=1):
        # 1) get exactly one feature row for this (user, recipe)
        X1 = get_feature_rows(user_id, [recipe_id], X_all, feature_names)
        if len(X1) != 1:
            # if duplicates exist, keep first; or raise if you prefer
            X1 = X1.iloc[[0]]

        # 2) compute shap values for a single row
        shap_vals = shap_explainer.shap_values(X1)

        # shap can return list for multiclass; handle both
        if isinstance(shap_vals, list):
            # pick class 1 if binary; otherwise pick argmax prediction, etc.
            shap_vals_1 = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_vals_1 = shap_vals

        # 3) build modern SHAP Explanation object (works great for plots)
        base_value = getattr(shap_explainer, "expected_value", 0.0)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        exp = shap.Explanation(
            values=shap_vals_1[0],
            base_values=base_value,
            data=X1.iloc[0].values,
            feature_names=feature_names,
        )

        # 4) save plots (waterfall + bar)
        # Waterfall
        plt.figure()
        shap.plots.waterfall(exp, max_display=top_k_features, show=False)
        wf_path = out_dir / f"{rank:02d}_{recipe_id}_waterfall.png"
        plt.tight_layout()
        plt.savefig(wf_path, dpi=160, bbox_inches="tight")
        plt.close()

        # Bar (global-ish view for this item)
        plt.figure()
        shap.plots.bar(exp, max_display=top_k_features, show=False)
        bar_path = out_dir / f"{rank:02d}_{recipe_id}_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=160, bbox_inches="tight")
        plt.close()

        results.append(
            {
                "recipe_id": recipe_id,
                "score": float(score),
                "waterfall_png": str(wf_path).replace("\\", "/"),
                "bar_png": str(bar_path).replace("\\", "/"),
            }
        )

    return results

# ---------- Endpoints ----------
@app.get("/users", response_model=list[UserOut])
def get_users(limit: int = Query(200, ge=1, le=5000)):
    """
    Returns a list of users (id + username).
    """
    return list_users(limit=limit)


@app.get("/recommend/{user_id}", response_model=RecommendOut)
def get_recommendations(user_id: str, n: int = Query(5, ge=1, le=50)):
    """
    Returns top-N recommendations for a given user.
    """
    users_df = load_users(Path("data") / "users.csv")
    recipes_df = load_recipes(Path("data") / "recipes.csv")
    recommender = Recommender.from_hybrid_checkpoints(
        retrieval_checkpoint_path="runs/retrieval/baseline/retrieval_final.pt",
        ranker_model_path="runs/ranking/baseline/ranker.pkl",
        embeddings_path="runs/retrieval/baseline/recipe_embeddings.npy",
        users_df=users_df,
        recipes_df=recipes_df,
        device="cpu",
    )

    shap_explainer = shap.TreeExplainer(recommender.model.ranking_model.model)

    t0 = time.perf_counter()

    try:
        recs = recommender.recommend(user_id, n=n)

    except KeyError:
        raise HTTPException(status_code=404, detail="user_id not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    plots = save_shap_plots_per_recommendation(
            user_id=user_id,
            recs=recs,
            shap_explainer=shap_explainer,
            X_all=X_ALL,
            feature_names=FEATURE_NAMES,
            out_dir=ART,
            top_k_features=15,
        )

    took_ms = (time.perf_counter() - t0) * 1000.0
    recs_info = get_info(recs)
    return RecommendOut(user_id=user_id, recommendations=recs_info, took_ms=took_ms)


@app.get("/", response_class=HTMLResponse)
def demo_page():
    """
    Serves demo UI page.
    """
    BASE_DIR = Path(__file__).parent
    UI_FILE = BASE_DIR / "templates/ui.html"
    if not UI_FILE.exists():
        return HTMLResponse("<h2>ui.html not found</h2>", status_code=404)

    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))
