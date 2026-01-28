from pathlib import Path
import time
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from coffee_recipe_recommender.inference.recommender import Recommender
from coffee_recipe_recommender.training.loaders import load_interactions, load_recipes, load_users
from src.client.services.recommend_service import RecommendOut, get_info, recommend
from src.client.services.users_service import UserOut, list_users


app = FastAPI(title="Coffee Recommender API", version="1.0")
app.mount("/images", StaticFiles(directory="data/images"), name="images")


# ---------- Endpoints ----------
@app.get("/users", response_model=list[UserOut])
def get_users(limit: int = Query(200, ge=1, le=5000)):
    """
    Returns a list of users (id + username).
    """
    return list_users(limit=limit)

# --- UI-friendly SHAP explanations (no plots) ---
EQUIPMENT_FEATURES = {
    "equipment_match",
    "equipment_coverage",
    "equipment_missing_count",
    "equipment_sophistication_match",
    "equipment_enables_difficulty",
    "equipment_specialty_affinity",
    "equipment_taste_squared",
    "has_espresso_machine",
    "has_grinder",
    "has_milk_frother",
    "requires_espresso_machine",
    "requires_milk_frother",
}

FEATURE_GROUPS = {
    "Taste match": [
        "taste_cosine_similarity",
        "taste_weighted_similarity",
        "taste_diff_mean",
        "taste_manhattan_distance",
        "taste_euclidean_distance",
        "taste_complexity_match",
        "dominant_taste_alignment",
        "taste_match_squared",
        "reliable_taste_match",
        "sweet_preference_match",
        "strong_tag_strength_match",
        "quality_fit_score",
    ],
    "Fits your habits": [
        "prep_time_acceptable",
        "prep_time_ratio",
        "prep_time_comfort",
        "portion_size_match",
        "portion_size_ratio",
        "portion_alignment_x_completion",
        "consumed_portion_match",
        "consumed_strength_match",
        "consumed_difficulty_match",
        "strength_consistency_score",
        "classic_consistency_score",
        "overall_alignment_score",
        "triple_alignment_score",
    ],
    "Time & context": [
        "morning_strength_score",
        "evening_specialty_score",
        "weekend_project_score",
        "hot_morning_score",
        "cold_afternoon_score",
        "quick_activity_score",
        "morning_combo_score",
        "afternoon_combo_score",
        "evening_combo_score",
        "weekend_exploration_score",
    ],
    "Discovery & novelty": [
        "exploration_novelty_score",
        "experienced_explorer",
        "user_exploration_ratio",
        "expert_rarity_affinity",
        "beginner_popular_affinity",
    ],
    "Quality & popularity": [
        "recipe_popularity_score",
        "popular_aligned_score",
        "popularity_momentum",
        "recipe_global_popularity",
        "recipe_category_popularity",
        "picky_user_good_recipe",
    ],
    "Dietary compatibility": [
        "dietary_compatible",
        "vegan_compatible_taste",
        "restricted_simple_score",
        "requires_milk",
    ],
}

GROUP_TEMPLATES = {
    "Taste match": "Matches your taste preferences",
    "Equipment fit": "Fits your equipment",
    "Fits your habits": "Fits how you usually make coffee",
    "Time & context": "Good choice for this time/situation",
    "Discovery & novelty": "Balanced discovery vs familiar",
    "Quality & popularity": "Trusted / popular choice",
    "Dietary compatibility": "Fits your dietary preferences",
}

def _shap_mat_and_base(shap_values, expected_value):
    # shap_values can be ndarray or list (classification)
    if isinstance(shap_values, list):
        shap_mat = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        base = expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1] if len(base) > 1 else base[0]
    else:
        shap_mat = shap_values
        base = expected_value
    return np.asarray(shap_mat), float(base)

def _group_contribs(shap_row: np.ndarray, feature_names: list[str]) -> pd.Series:
    s = pd.Series(shap_row, index=feature_names)

    grouped = {}
    for group, feats in FEATURE_GROUPS.items():
        present = [f for f in feats if f in s.index]
        grouped[group] = float(s[present].sum()) if present else 0.0

    # sort by absolute impact
    return pd.Series(grouped).sort_values(key=lambda x: np.abs(x), ascending=False)

def explain_for_ui(
    *,
    recipe_id: str,
    shap_row: np.ndarray,
    feature_names: list[str],
    base: float,
    pred: float,
    max_reasons: int = 3,
    max_tradeoffs: int = 1,
) -> dict:
    s = pd.Series(shap_row, index=feature_names)

    # --- 1) move equipment contribution into baseline ---
    equip_contrib = s[list(EQUIPMENT_FEATURES & set(s.index))].sum()
    base_adj = base + float(equip_contrib)

    # zero-out equipment so it never appears in reasons
    s.loc[list(EQUIPMENT_FEATURES & set(s.index))] = 0.0

    # --- 2) group remaining SHAP values ---
    grouped = {}
    for group, feats in FEATURE_GROUPS.items():
        present = [f for f in feats if f in s.index]
        grouped[group] = float(s[present].sum()) if present else 0.0

    grouped = pd.Series(grouped).sort_values(key=lambda x: abs(x), ascending=False)

    positives = grouped[grouped > 0].head(max_reasons)
    negatives = grouped[grouped < 0].head(max_tradeoffs)

    return {
        "recipe_id": recipe_id,
        # lift relative to "equipment-ok" baseline
        "score_lift": float(pred - base_adj),
        "reasons": [
            {"label": GROUP_TEMPLATES[g], "impact": float(v)}
            for g, v in positives.items()
        ],
        "tradeoffs": [
            {"label": GROUP_TEMPLATES[g], "impact": float(v)}
            for g, v in negatives.items()
        ],
    }

@app.get("/recommend/{user_id}", response_model=RecommendOut)
def get_recommendations(user_id: str, n: int = Query(5, ge=1, le=50)):
    users_df = load_users(Path("data") / "users.csv")
    recipes_df = load_recipes(Path("data") / "recipes.csv")
    train_df = load_interactions(Path("data") / "interactions_train_split.csv")

    recommender = Recommender.from_hybrid_checkpoints(
        retrieval_checkpoint_path="runs/retrieval/baseline/retrieval_final.pt",
        # ranker_model_path="runs/ranking/improved-features/ranker.pkl",
        ranker_model_path="runs/ranking/very-advanced-features/ranker.pkl",
        vector_store_path="data/chroma",
        # feature_store_path="data/legacy_feature_store.db",
        feature_store_path="data/feature_store.db",
        users_df=users_df,
        recipes_df=recipes_df,
        device="cpu",
        # preset="legacy"
    )

    t0 = time.perf_counter()

    try:
        # returns top list, feature rows for top, shap values for top, and explainer
        t0 = time.perf_counter()
        top, X_top, shap_values, explainer = recommender.recommend_with_shap(
            user_id, users_df, recipes_df, train_df, n=n
        )
        took_ms = (time.perf_counter() - t0) * 1000.0

        # get raw model predictions for these same rows (for lift)
        preds = recommender.model.ranking_model.predict(X_top)  # <-- adjust if your attribute differs

        shap_mat, base = _shap_mat_and_base(shap_values, explainer.expected_value)
        feature_names = X_top.columns.tolist()

        # build per-recipe UI explanations
        ui_by_rid = {}
        for i, (rid, score) in enumerate(top):
            ui_by_rid[rid] = explain_for_ui(
                recipe_id=rid,
                shap_row=shap_mat[i],
                feature_names=feature_names,
                base=base,
                pred=float(preds[i]),
                max_reasons=3,
                max_tradeoffs=1,
            )

        # your existing enrichment (name/desc/tags/etc.)
        recs_info = get_info(top)

        # attach UI explanation into each recommendation dict
        for r in recs_info:
            rid = r.get("recipe_id")
            if rid in ui_by_rid:
                r["why_recommended"] = ui_by_rid[rid]

    except KeyError:
        raise HTTPException(status_code=404, detail="user_id not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
