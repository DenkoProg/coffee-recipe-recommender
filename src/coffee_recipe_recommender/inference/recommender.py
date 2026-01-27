import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from coffee_recipe_recommender.models.hybrid import HybridRecommenderModel
from coffee_recipe_recommender.models.ranking import LightGBMRankerModel
from coffee_recipe_recommender.models.retrieval import RetrievalRecommenderModel, TwoTowerModel


class Recommender:
    """
    Generic recommender that works with any underlying model.

    Supports:
    - Retrieval-only (Two-Tower)
    - Hybrid (Two-Tower + LightGBM)
    - Any future model implementing predict(user_id, n, exclude) method
    """

    def __init__(self, model: Any):
        """
        Initialize generic recommender.

        Args:
            model: Any model with a predict(user_id, n, exclude_recipes) method
                   Examples: RetrievalRecommenderModel, HybridRecommenderModel
        """
        self.model = model

    def recommend(
        self,
        user_id: str,
        n: int = 5,
        exclude_recipes: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Generate top-N recommendations.

        Args:
            user_id: User identifier
            n: Number of recommendations
            exclude_recipes: Optional recipes to exclude

        Returns:
            List of (recipe_id, score) tuples
        """
        if hasattr(self.model, "predict"):
            return self.model.predict(user_id, n, exclude_recipes)
        elif hasattr(self.model, "recommend"):
            return self.model.recommend(user_id, n, exclude_recipes)
        else:
            raise AttributeError("Model must have either 'predict' or 'recommend' method")

    def recommend_batch(
        self,
        user_ids: list[str],
        n: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Generate recommendations for multiple users."""
        recommendations = {}
        for user_id in user_ids:
            try:
                recommendations[user_id] = self.recommend(user_id, n)
            except (ValueError, KeyError):
                continue
        return recommendations

    @classmethod
    def from_retrieval_checkpoint(
        cls,
        checkpoint_path: Path,
        embeddings_path: Path,
        users_df: pd.DataFrame | None = None,
        device: str = "cpu",
    ) -> "Recommender":
        """Load retrieval-only recommender from checkpoint."""
        with torch.serialization.safe_globals([pathlib.PosixPath]):
            checkpoint = torch.load(checkpoint_path, map_location=device)

        model_args = checkpoint["args"]

        model = TwoTowerModel(
            num_users=len(checkpoint["user_to_idx"]),
            num_recipes=len(checkpoint["recipe_to_idx"]),
            embedding_dim=model_args["embedding_dim"],
            hidden_dims=model_args["hidden_dims"],
            use_features=model_args["use_features"],
            dropout=model_args.get("dropout", 0.2),
            temperature=model_args.get("temperature", 0.07),
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        recipe_embeddings = np.load(embeddings_path)

        retrieval_model = RetrievalRecommenderModel(
            model=model,
            recipe_embeddings=recipe_embeddings,
            user_to_idx=checkpoint["user_to_idx"],
            recipe_to_idx=checkpoint["recipe_to_idx"],
            idx_to_recipe=checkpoint["idx_to_recipe"],
            users_df=users_df,
            device=device,
        )

        return cls(model=retrieval_model)

    @classmethod
    def from_hybrid_checkpoints(
        cls,
        retrieval_checkpoint_path: Path,
        ranker_model_path: Path,
        embeddings_path: Path,
        users_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
        candidate_size: int = 50,
        device: str = "cpu",
    ) -> "Recommender":
        """Load hybrid recommender from retrieval + ranking checkpoints."""
        with torch.serialization.safe_globals([pathlib.PosixPath]):
            checkpoint = torch.load(retrieval_checkpoint_path, map_location=device)

        model_args = checkpoint["args"]

        retrieval_model = TwoTowerModel(
            num_users=len(checkpoint["user_to_idx"]),
            num_recipes=len(checkpoint["recipe_to_idx"]),
            embedding_dim=model_args["embedding_dim"],
            hidden_dims=model_args["hidden_dims"],
            use_features=model_args["use_features"],
            dropout=model_args.get("dropout", 0.2),
            temperature=model_args.get("temperature", 0.07),
        )
        retrieval_model.load_state_dict(checkpoint["model_state_dict"])

        ranking_model = LightGBMRankerModel()
        ranking_model.load(ranker_model_path)

        recipe_embeddings = np.load(embeddings_path)

        hybrid = HybridRecommenderModel(
            retrieval_model=retrieval_model,
            ranking_model=ranking_model,
            recipe_embeddings=recipe_embeddings,
            user_to_idx=checkpoint["user_to_idx"],
            recipe_to_idx=checkpoint["recipe_to_idx"],
            idx_to_recipe=checkpoint["idx_to_recipe"],
            users_df=users_df,
            recipes_df=recipes_df,
            candidate_size=candidate_size,
            device=device,
        )

        return cls(model=hybrid)
