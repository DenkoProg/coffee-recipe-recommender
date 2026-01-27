from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    """
    Dataset for Two-Tower retrieval model training with in-batch negative sampling.

    Returns (user_idx, recipe_idx) pairs where positive pairs come from actual
    interactions. Negative sampling is handled by InfoNCE loss using in-batch negatives.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_to_idx: dict[str, int],
        recipe_to_idx: dict[str, int],
        min_rating: float = 3.5,
        use_completed_only: bool = False,
    ):
        """
        Initialize retrieval dataset.

        Args:
            interactions_df: DataFrame with user_id, recipe_id, rating, completed columns
            user_to_idx: Mapping from user_id string to integer index
            recipe_to_idx: Mapping from recipe_id string to integer index
            min_rating: Minimum rating to consider as positive interaction
            use_completed_only: If True, only use completed=True interactions
        """
        self.user_to_idx = user_to_idx
        self.recipe_to_idx = recipe_to_idx

        df = interactions_df.copy()

        if use_completed_only:
            df = df[df["completed"] == True]  # noqa: E712

        df["is_positive"] = (df["rating"].fillna(0) >= min_rating) | (df["completed"] == True)  # noqa: E712
        df = df[df["is_positive"]]

        df["user_idx"] = df["user_id"].map(user_to_idx)
        df["recipe_idx"] = df["recipe_id"].map(recipe_to_idx)
        df = df.dropna(subset=["user_idx", "recipe_idx"])

        self.user_indices = df["user_idx"].astype(int).values
        self.recipe_indices = df["recipe_idx"].astype(int).values
        self.num_users = len(user_to_idx)
        self.num_recipes = len(recipe_to_idx)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single positive (user, recipe) pair.

        Returns:
            (user_idx, recipe_idx) as torch.LongTensor
        """
        user_idx = self.user_indices[idx]
        recipe_idx = self.recipe_indices[idx]
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(recipe_idx, dtype=torch.long)


class RetrievalDatasetWithFeatures(Dataset):
    """
    Extended retrieval dataset that includes user and recipe features.

    Useful for cold-start scenarios and feature-enhanced towers.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
        user_to_idx: dict[str, int],
        recipe_to_idx: dict[str, int],
        min_rating: float = 3.5,
        use_completed_only: bool = False,
    ):
        """
        Initialize retrieval dataset with features.

        Args:
            interactions_df: Interactions DataFrame
            users_df: Users DataFrame with features
            recipes_df: Recipes DataFrame with features
            user_to_idx: User ID to index mapping
            recipe_to_idx: Recipe ID to index mapping
            min_rating: Minimum rating for positive interaction
            use_completed_only: Only use completed interactions
        """
        # Use base dataset for filtering
        base_dataset = RetrievalDataset(interactions_df, user_to_idx, recipe_to_idx, min_rating, use_completed_only)
        self.user_indices = base_dataset.user_indices
        self.recipe_indices = base_dataset.recipe_indices
        self.num_users = base_dataset.num_users
        self.num_recipes = base_dataset.num_recipes

        self.user_features = self._extract_user_features(users_df, user_to_idx)
        self.recipe_features = self._extract_recipe_features(recipes_df, recipe_to_idx)

    def _extract_user_features(self, users_df: pd.DataFrame, user_to_idx: dict[str, int]) -> np.ndarray:
        """Extract user taste preferences as feature matrix (n_users, 4)."""
        # Sort by index to ensure alignment
        sorted_users = sorted(user_to_idx.items(), key=lambda x: x[1])
        user_ids = [uid for uid, _ in sorted_users]

        # Create feature matrix
        df_sorted = users_df.set_index("user_id").loc[user_ids]

        features = df_sorted[
            ["taste_pref_bitterness", "taste_pref_sweetness", "taste_pref_acidity", "taste_pref_body"]
        ].values

        return features.astype(np.float32)

    def _extract_recipe_features(self, recipes_df: pd.DataFrame, recipe_to_idx: dict[str, int]) -> np.ndarray:
        """Extract recipe taste profiles as feature matrix (n_recipes, 4)."""
        # Sort by index
        sorted_recipes = sorted(recipe_to_idx.items(), key=lambda x: x[1])
        recipe_ids = [rid for rid, _ in sorted_recipes]

        df_sorted = recipes_df.set_index("recipe_id").loc[recipe_ids]

        features = df_sorted[["taste_bitterness", "taste_sweetness", "taste_acidity", "taste_body"]].values

        return features.astype(np.float32)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single training sample with features.

        Returns:
            Dictionary with keys:
                - user_idx: User index (long)
                - recipe_idx: Recipe index (long)
                - user_features: User taste preferences (float, shape=(4,))
                - recipe_features: Recipe taste profile (float, shape=(4,))
        """
        user_idx = self.user_indices[idx]
        recipe_idx = self.recipe_indices[idx]

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "recipe_idx": torch.tensor(recipe_idx, dtype=torch.long),
            "user_features": torch.from_numpy(self.user_features[user_idx]).float(),
            "recipe_features": torch.from_numpy(self.recipe_features[recipe_idx]).float(),
        }
