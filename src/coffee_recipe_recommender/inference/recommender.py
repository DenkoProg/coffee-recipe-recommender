import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from coffee_recipe_recommender.models.hybrid import HybridRecommenderModel
from coffee_recipe_recommender.models.ranking import LightGBMRankerModel
from coffee_recipe_recommender.models.retrieval import ColdStartEncoder, RetrievalRecommenderModel, TwoTowerModel


# Resolve default data path relative to this file to handle calls from subdirectories
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "data" / "chroma"


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
        users_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Generate top-N recipe recommendations for a user.

        Args:
            user_id: Target user identifier
            users_df: Users dataframe (users.csv loaded)
            recipes_df: Recipes dataframe (recipes.csv loaded)
            train_df: Training interactions (interactions_train.csv loaded)
            n: Number of recommendations to return

        Returns:
            List of (recipe_id, score) tuples, sorted by score descending.
        """
        if hasattr(self.model, "recommend"):
            return self.model.recommend(user_id, users_df, recipes_df, train_df, n)
        else:
            raise AttributeError("Model must have 'recommend' method")

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
        embeddings_path: Path | None = None,
        users_df: pd.DataFrame | None = None,
        cold_start_path: Path | None = None,
        device: str = "cpu",
        vector_store_path: Path | str | None = DEFAULT_VECTOR_STORE,
    ) -> "Recommender":
        """Load retrieval-only recommender from checkpoint.

        Args:
            checkpoint_path: Path to two-tower model checkpoint
            embeddings_path: Path to .npy embeddings file (optional if vector_store_path provided)
            users_df: Optional users dataframe
            cold_start_path: Optional path to cold-start encoder
            device: Device for inference
            vector_store_path: Optional path to ChromaDB vector store (alternative to embeddings_path)
        """
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

        # Load embeddings from VectorStore or .npy file
        # Load embeddings from VectorStore or .npy file
        # Prioritize vector_store_path (default) -> embeddings_path
        recipe_embeddings = None

        # Try loading from vector store first
        if vector_store_path is not None:
            # Convert to Path if string
            vs_path = Path(vector_store_path)
            if vs_path.exists():
                from coffee_recipe_recommender.db.vector_store import VectorStore

                store = VectorStore(vs_path)
                if store.exists():
                    recipe_embeddings, _, _ = store.load_embeddings()

        # Fallback to embeddings_path if vector store didn't work
        if recipe_embeddings is None and embeddings_path is not None:
            recipe_embeddings = np.load(embeddings_path)

        if recipe_embeddings is None:
            raise ValueError(
                f"Could not load embeddings. vector_store_path={vector_store_path} "
                f"(exists={Path(vector_store_path).exists() if vector_store_path else False}), "
                f"embeddings_path={embeddings_path}"
            )

        retrieval_model = RetrievalRecommenderModel(
            model=model,
            recipe_embeddings=recipe_embeddings,
            user_to_idx=checkpoint["user_to_idx"],
            recipe_to_idx=checkpoint["recipe_to_idx"],
            idx_to_recipe=checkpoint["idx_to_recipe"],
            users_df=users_df,
            device=device,
        )

        # Optionally load cold-start encoder and attach to retrieval wrapper
        if cold_start_path is not None:
            cs_ckpt = torch.load(cold_start_path, map_location=device)
            feature_dim = cs_ckpt.get("feature_dim", 4)
            encoder = ColdStartEncoder(feature_dim=feature_dim, embedding_dim=model.embedding_dim)
            encoder.load_state_dict(cs_ckpt["state_dict"])
            encoder.to(device).eval()
            retrieval_model.cold_start_encoder = encoder

        return cls(model=retrieval_model)

    @classmethod
    def from_hybrid_checkpoints(
        cls,
        retrieval_checkpoint_path: Path,
        ranker_model_path: Path,
        embeddings_path: Path | None = None,
        users_df: pd.DataFrame = None,
        recipes_df: pd.DataFrame = None,
        candidate_size: int = 50,
        device: str = "cpu",
        cold_start_path: Path | None = None,
        feature_store_path: Path | str | None = None,
        feature_subset: list[str] | None = None,
        vector_store_path: Path | str | None = DEFAULT_VECTOR_STORE,
    ) -> "Recommender":
        """Load hybrid recommender from retrieval + ranking checkpoints.

        Args:
            retrieval_checkpoint_path: Path to two-tower model checkpoint
            ranker_model_path: Path to LightGBM ranker model
            embeddings_path: Path to .npy embeddings file (optional if vector_store_path provided)
            users_df: Users dataframe
            recipes_df: Recipes dataframe
            candidate_size: Number of candidates for retrieval stage
            device: Device for inference
            cold_start_path: Optional path to cold-start encoder
            feature_store_path: Optional path to SQLite feature store
            feature_subset: Optional list of features to use
            vector_store_path: Optional path to ChromaDB vector store (alternative to embeddings_path)
        """
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

        # Load embeddings from VectorStore or .npy file
        # Load embeddings from VectorStore or .npy file
        # Prioritize vector_store_path (default) -> embeddings_path
        recipe_embeddings = None

        # Try loading from vector store first
        if vector_store_path is not None:
            # Convert to Path if string
            vs_path = Path(vector_store_path)
            if vs_path.exists():
                from coffee_recipe_recommender.db.vector_store import VectorStore

                store = VectorStore(vs_path)
                if store.exists():
                    recipe_embeddings, _, _ = store.load_embeddings()

        # Fallback to embeddings_path if vector store didn't work
        if recipe_embeddings is None and embeddings_path is not None:
            recipe_embeddings = np.load(embeddings_path)

        if recipe_embeddings is None:
            raise ValueError(
                f"Could not load embeddings. vector_store_path={vector_store_path} "
                f"(exists={Path(vector_store_path).exists() if vector_store_path else False}), "
                f"embeddings_path={embeddings_path}"
            )

        # Optionally load cold-start encoder
        cs_encoder = None
        if cold_start_path is not None:
            cs_ckpt = torch.load(cold_start_path, map_location=device)
            feature_dim = cs_ckpt.get("feature_dim", 4)
            cs_encoder = ColdStartEncoder(feature_dim=feature_dim, embedding_dim=retrieval_model.embedding_dim)
            cs_encoder.load_state_dict(cs_ckpt["state_dict"])
            cs_encoder.to(device).eval()

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
            cold_start_encoder=cs_encoder,
            feature_store_path=feature_store_path,
            feature_subset=feature_subset,
        )

        return cls(model=hybrid)
