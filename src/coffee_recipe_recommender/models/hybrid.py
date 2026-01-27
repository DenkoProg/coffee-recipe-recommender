from typing import Any

import numpy as np
import pandas as pd
import torch

from coffee_recipe_recommender.models.ranking import LightGBMRankerModel
from coffee_recipe_recommender.models.retrieval import TwoTowerModel
from coffee_recipe_recommender.preprocessing.preprocessing import FeatureEngineer


class HybridRecommenderModel:
    """
    Two-stage hybrid recommender: retrieval + reranking.

    Stage 1 (Retrieval): Two-Tower model generates K candidates (~100)
    Stage 2 (Ranking): LightGBM reranks candidates using rich features
    """

    def __init__(
        self,
        retrieval_model: TwoTowerModel,
        ranking_model: LightGBMRankerModel,
        recipe_embeddings: np.ndarray,
        user_to_idx: dict[str, int],
        recipe_to_idx: dict[str, int],
        idx_to_recipe: dict[int, str],
        users_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
        candidate_size: int = 100,
        device: str = "cpu",
        cold_start_encoder: Any = None,
    ):
        """
        Initialize hybrid model.

        Args:
            retrieval_model: Trained Two-Tower model
            ranking_model: Trained LightGBM ranker
            recipe_embeddings: Pre-computed recipe embeddings
            user_to_idx: User ID to index mapping
            recipe_to_idx: Recipe ID to index mapping
            idx_to_recipe: Index to recipe ID mapping
            users_df: Users DataFrame with features
            recipes_df: Recipes DataFrame with features
            candidate_size: Number of candidates from retrieval stage
            device: Device for inference
        """
        self.retrieval_model = retrieval_model.to(device).eval()
        self.ranking_model = ranking_model
        self.recipe_embeddings = torch.from_numpy(recipe_embeddings).float().to(device)
        self.user_to_idx = user_to_idx
        self.recipe_to_idx = recipe_to_idx
        self.idx_to_recipe = idx_to_recipe
        self.users_df = users_df
        self.recipes_df = recipes_df
        self.candidate_size = candidate_size
        self.device = device
        self.use_features = retrieval_model.user_tower.use_features
        self.feature_engineer = FeatureEngineer()
        self.cold_start_encoder = cold_start_encoder.to(device) if cold_start_encoder is not None else None

    def _get_equipment_compatible_recipes(self, user_id: str) -> set[int]:
        """Get set of recipe indices compatible with user's owned equipment.

        Args:
            user_id: User identifier

        Returns:
            Set of recipe indices that have required_equipment as subset of user's owned_equipment.
            If no equipment constraint, returns all indices.
        """
        if self.users_df is None:
            return set(range(len(self.idx_to_recipe)))

        user_rows = self.users_df[self.users_df["user_id"] == user_id]
        if user_rows.empty:
            return set(range(len(self.idx_to_recipe)))

        user_equipment = set(user_rows.iloc[0].get("owned_equipment") or [])
        if not user_equipment or self.recipes_df is None:
            return set(range(len(self.idx_to_recipe)))  # No equipment constraint

        compatible_indices = set()
        for idx, recipe_id in self.idx_to_recipe.items():
            recipe_rows = self.recipes_df[self.recipes_df["recipe_id"] == recipe_id]
            if not recipe_rows.empty:
                required_eq = set(recipe_rows.iloc[0].get("required_equipment") or [])
                if required_eq.issubset(user_equipment):
                    compatible_indices.add(idx)
            else:
                compatible_indices.add(idx)  # Unknown recipe - include by default

        return compatible_indices if compatible_indices else set(range(len(self.idx_to_recipe)))

    @torch.no_grad()
    def get_candidates(
        self,
        user_id: str,
        k: int,
        exclude_recipes: set[str] | None = None,
    ) -> list[str]:
        """
        Stage 1: Get top-K candidates from retrieval model.

        Args:
            user_id: User identifier
            k: Number of candidates to retrieve
            exclude_recipes: Optional recipes to exclude

        Returns:
            List of candidate recipe IDs
        """
        # Support cold-start users via optional `cold_start_encoder`.
        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)

            user_features = None
            if self.use_features:
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

            user_emb = self.retrieval_model.get_user_embeddings(user_tensor, user_features)
        else:
            if self.cold_start_encoder is None or self.users_df is None:
                raise ValueError(f"Unknown user_id: {user_id}")
            user_row = self.users_df[self.users_df["user_id"] == user_id]
            if user_row.empty:
                raise ValueError(f"Unknown user_id: {user_id}")
            user_row = user_row.iloc[0]
            cs_feat = torch.tensor(
                [
                    [
                        user_row.get("taste_pref_bitterness", 0.5),
                        user_row.get("taste_pref_sweetness", 0.5),
                        user_row.get("taste_pref_acidity", 0.5),
                        user_row.get("taste_pref_body", 0.5),
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            )
            user_emb = self.cold_start_encoder(cs_feat)
        similarities = torch.matmul(user_emb, self.recipe_embeddings.T).squeeze(0)

        # Apply hard equipment filtering
        compatible = self._get_equipment_compatible_recipes(user_id)
        for idx in range(len(similarities)):
            if idx not in compatible:
                similarities[idx] = -float("inf")

        if exclude_recipes:
            exclude_indices = [self.recipe_to_idx[rid] for rid in exclude_recipes if rid in self.recipe_to_idx]
            if exclude_indices:
                similarities[exclude_indices] = -float("inf")

        top_indices = torch.topk(similarities, k=min(k, len(similarities))).indices
        candidates = [self.idx_to_recipe[idx.item()] for idx in top_indices]

        return candidates

    def extract_features(
        self,
        user_id: str,
        recipe_id: str,
    ) -> dict[str, float]:
        """
        Extract ranking features for a (user, recipe) pair.

        Args:
            user_id: User identifier
            recipe_id: Recipe identifier

        Returns:
            Dictionary of feature name to value
        """
        user_row = self.users_df[self.users_df["user_id"] == user_id].iloc[0]
        recipe_row = self.recipes_df[self.recipes_df["recipe_id"] == recipe_id].iloc[0]

        features = {}

        # Taste similarity (cosine)
        user_taste = np.array(
            [
                user_row["taste_pref_bitterness"],
                user_row["taste_pref_sweetness"],
                user_row["taste_pref_acidity"],
                user_row["taste_pref_body"],
            ]
        )
        recipe_taste = np.array(
            [
                recipe_row["taste_bitterness"],
                recipe_row["taste_sweetness"],
                recipe_row["taste_acidity"],
                recipe_row["taste_body"],
            ]
        )
        features["taste_cosine"] = float(
            np.dot(user_taste, recipe_taste) / (np.linalg.norm(user_taste) * np.linalg.norm(recipe_taste) + 1e-8)
        )
        features["taste_l2_dist"] = float(np.linalg.norm(user_taste - recipe_taste))

        # Equipment compatibility
        user_equipment = set(user_row.get("owned_equipment") or [])
        recipe_equipment = set(recipe_row.get("required_equipment") or [])
        features["equipment_overlap"] = len(user_equipment & recipe_equipment)
        features["equipment_missing"] = len(recipe_equipment - user_equipment)
        features["has_all_equipment"] = 1.0 if recipe_equipment.issubset(user_equipment) else 0.0

        # Recipe attributes
        features["strength"] = float(recipe_row.get("strength", 0))
        features["portion_size_ml"] = float(recipe_row.get("portion_size_ml", 0))
        features["preparation_time_minutes"] = float(recipe_row.get("preparation_time_minutes", 0))

        # Difficulty encoding
        difficulty = recipe_row.get("difficulty", "").lower()
        features["difficulty_beginner"] = 1.0 if difficulty == "beginner" else 0.0
        features["difficulty_intermediate"] = 1.0 if difficulty == "intermediate" else 0.0
        features["difficulty_advanced"] = 1.0 if difficulty == "advanced" else 0.0

        # User preferences match
        pref_strength = user_row.get("preferred_strength", 3)
        features["strength_match"] = 1.0 if abs(features["strength"] - pref_strength) <= 1 else 0.0

        pref_portion = user_row.get("preferred_portion_size", "medium").lower()
        portion_ml = features["portion_size_ml"]
        if pref_portion == "small":
            features["portion_match"] = 1.0 if portion_ml <= 150 else 0.0
        elif pref_portion == "medium":
            features["portion_match"] = 1.0 if 150 < portion_ml <= 300 else 0.0
        else:
            features["portion_match"] = 1.0 if portion_ml > 300 else 0.0

        return features

    def rank_candidates(
        self,
        user_id: str,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """
        Stage 2: Rerank candidates using LightGBM.

        Args:
            user_id: User identifier
            candidates: List of candidate recipe IDs

        Returns:
            List of (recipe_id, score) tuples sorted by score descending
        """
        if not candidates:
            return []

        candidates_df = pd.DataFrame(
            {
                "user_id": [user_id] * len(candidates),
                "recipe_id": candidates,
            }
        )
        features_df = self.feature_engineer.generate(candidates_df, self.users_df, self.recipes_df)
        scores = self.ranking_model.predict(features_df)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return ranked

    def recommend(
        self,
        user_id: str,
        users_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Full two-stage recommendation: retrieval → ranking.

        Args:
            user_id: Target user identifier
            users_df: Users dataframe (users.csv loaded)
            recipes_df: Recipes dataframe (recipes.csv loaded)
            train_df: Training interactions (interactions_train.csv loaded)
            n: Number of recommendations to return

        Returns:
            List of (recipe_id, score) tuples, sorted by score descending.
        """
        # Stage 1: Retrieve candidates
        candidates = self.get_candidates(user_id, k=self.candidate_size)

        if not candidates:
            return []

        # Stage 2: Rerank
        ranked = self.rank_candidates(user_id, candidates)

        return ranked[:n]
