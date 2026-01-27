import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from coffee_recipe_recommender.models.retrieval import TwoTowerModel


class RetrievalRecommender:
    """
    Fast retrieval-based recommender using pre-computed recipe embeddings.

    Uses cosine similarity between user embedding and all recipe embeddings
    for efficient candidate generation.
    """

    def __init__(
        self,
        model: TwoTowerModel,
        recipe_embeddings: np.ndarray,
        user_to_idx: dict[str, int],
        recipe_to_idx: dict[str, int],
        idx_to_recipe: dict[int, str],
        users_df: pd.DataFrame | None = None,
        device: str = "cpu",
    ):
        """
        Initialize retrieval recommender.

        Args:
            model: Trained TwoTowerModel
            recipe_embeddings: Pre-computed recipe embeddings (n_recipes, embedding_dim)
            user_to_idx: User ID to index mapping
            recipe_to_idx: Recipe ID to index mapping
            idx_to_recipe: Index to recipe ID mapping
            users_df: Optional users DataFrame for features
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()

        self.recipe_embeddings = torch.from_numpy(recipe_embeddings).float().to(device)
        self.user_to_idx = user_to_idx
        self.recipe_to_idx = recipe_to_idx
        self.idx_to_recipe = idx_to_recipe
        self.users_df = users_df
        self.device = device
        self.use_features = model.user_tower.use_features

    @torch.no_grad()
    def recommend(
        self,
        user_id: str,
        n: int = 5,
        exclude_recipes: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: User identifier
            n: Number of recommendations to return
            exclude_recipes: Optional set of recipe IDs to exclude (e.g., already tried)

        Returns:
            List of (recipe_id, score) tuples sorted by score descending
        """
        # Get user index
        if user_id not in self.user_to_idx:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)

        # Get user features if needed
        user_features = None
        if self.use_features and self.users_df is not None:
            user_row = self.users_df[self.users_df["user_id"] == user_id].iloc[0]
            user_features = torch.tensor(
                [
                    [
                        user_row["taste_pref_bitterness"],
                        user_row["taste_pref_sweetness"],
                        user_row["taste_pref_acidity"],
                        user_row["taste_pref_body"],
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            )

        # Get user embedding
        user_emb = self.model.get_user_embeddings(user_tensor, user_features)  # (1, embedding_dim)

        # Compute similarities with all recipes
        similarities = torch.matmul(user_emb, self.recipe_embeddings.T).squeeze(0)  # (n_recipes,)

        # Exclude recipes if specified
        if exclude_recipes:
            exclude_indices = [self.recipe_to_idx[rid] for rid in exclude_recipes if rid in self.recipe_to_idx]
            if exclude_indices:
                similarities[exclude_indices] = -float("inf")

        # Get top-N
        top_scores, top_indices = torch.topk(similarities, k=min(n, len(similarities)))

        recommendations = [
            (self.idx_to_recipe[idx.item()], score.item()) for idx, score in zip(top_indices, top_scores, strict=True)
        ]

        return recommendations

    @torch.no_grad()
    def recommend_batch(
        self,
        user_ids: list[str],
        n: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Generate recommendations for multiple users efficiently.

        Args:
            user_ids: List of user identifiers
            n: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (recipe_id, score) tuples
        """
        recommendations = {}

        for user_id in user_ids:
            try:
                recommendations[user_id] = self.recommend(user_id, n)
            except ValueError:
                # Skip unknown users
                continue

        return recommendations

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        embeddings_path: Path,
        users_df: pd.DataFrame | None = None,
        device: str = "cpu",
    ) -> "RetrievalRecommender":
        """
        Load recommender from checkpoint files.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            embeddings_path: Path to pre-computed recipe embeddings (.npy file)
            users_df: Optional users DataFrame for features
            device: Device for inference

        Returns:
            RetrievalRecommender instance
        """
        with torch.serialization.safe_globals([pathlib.PosixPath]):
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model arguments
        model_args = checkpoint["args"]

        # Initialize model
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

        # Load recipe embeddings
        recipe_embeddings = np.load(embeddings_path)

        return cls(
            model=model,
            recipe_embeddings=recipe_embeddings,
            user_to_idx=checkpoint["user_to_idx"],
            recipe_to_idx=checkpoint["recipe_to_idx"],
            idx_to_recipe=checkpoint["idx_to_recipe"],
            users_df=users_df,
            device=device,
        )
