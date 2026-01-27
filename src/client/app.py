from pathlib import Path
import time
from typing import List

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


@app.get("/recommend/{user_id}", response_model=RecommendOut)
def get_recommendations(user_id: str, n: int = Query(5, ge=1, le=50)):
    """
    Returns top-N recommendations for a given user.
    """
    users_df = load_users(Path("data") / "users.csv")
    recipes_df = load_recipes(Path("data") / "recipes.csv")
    recommender = Recommender.from_hybrid_checkpoints(
        retrieval_checkpoint_path="runs/retrieval/baseline/retrieval_final.pt",
        ranker_model_path="runs/ranking/improved-features/ranker.pkl",
        embeddings_path="runs/retrieval/baseline/recipe_embeddings.npy",
        users_df=users_df,
        recipes_df=recipes_df,
        device="cpu",
    )

    t0 = time.perf_counter()

    try:
        # Load training interactions (used for feature generation / context)
        train_df = load_interactions(Path("data") / "interactions_train_split.csv")
        recs = get_info(recommender.recommend(user_id, users_df, recipes_df, train_df, n=n))
    except KeyError:
        raise HTTPException(status_code=404, detail="user_id not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    took_ms = (time.perf_counter() - t0) * 1000.0
    return RecommendOut(user_id=user_id, recommendations=recs, took_ms=took_ms)


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
