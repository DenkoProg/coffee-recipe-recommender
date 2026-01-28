import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    User embedding tower.

    Combines learned user embedding with optional user features (taste preferences)
    and projects to a shared embedding space.
    """

    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        use_features: bool = False,
        feature_dim: int = 4,
        dropout: float = 0.2,
    ):
        """
        Initialize user tower.

        Args:
            num_users: Total number of users
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions (default [256, 128] before final 64)
            use_features: Whether to concatenate user features
            feature_dim: Dimension of user features (taste preferences)
            dropout: Dropout probability
        """
        super().__init__()

        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.use_features = use_features

        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        if hidden_dims is None:
            hidden_dims = [256, 128]
        input_dim = embedding_dim + (feature_dim if use_features else 0)
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_idx: torch.Tensor, user_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through user tower.

        Args:
            user_idx: User indices, shape (batch_size,)
            user_features: Optional user features, shape (batch_size, feature_dim)

        Returns:
            User embeddings, shape (batch_size, embedding_dim)
        """
        # Get user embedding
        user_emb = self.user_embedding(user_idx)  # (batch_size, embedding_dim)

        # Concatenate features if provided
        if self.use_features and user_features is not None:
            x = torch.cat([user_emb, user_features], dim=1)
        else:
            x = user_emb

        # Pass through MLP
        output = self.mlp(x)  # (batch_size, embedding_dim)

        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)

        return output


class RecipeTower(nn.Module):
    """
    Recipe embedding tower.

    Combines learned recipe embedding with optional recipe features (taste profile)
    and projects to a shared embedding space.
    """

    def __init__(
        self,
        num_recipes: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        use_features: bool = False,
        feature_dim: int = 4,
        dropout: float = 0.2,
    ):
        """
        Initialize recipe tower.

        Args:
            num_recipes: Total number of recipes
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
            use_features: Whether to concatenate recipe features
            feature_dim: Dimension of recipe features (taste profile)
            dropout: Dropout probability
        """
        super().__init__()

        self.num_recipes = num_recipes
        self.embedding_dim = embedding_dim
        self.use_features = use_features

        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)

        if hidden_dims is None:
            hidden_dims = [256, 128]
        input_dim = embedding_dim + (feature_dim if use_features else 0)
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.recipe_embedding.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, recipe_idx: torch.Tensor, recipe_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through recipe tower.

        Args:
            recipe_idx: Recipe indices, shape (batch_size,)
            recipe_features: Optional recipe features, shape (batch_size, feature_dim)

        Returns:
            Recipe embeddings, shape (batch_size, embedding_dim)
        """
        # Get recipe embedding
        recipe_emb = self.recipe_embedding(recipe_idx)  # (batch_size, embedding_dim)

        # Concatenate features if provided
        if self.use_features and recipe_features is not None:
            x = torch.cat([recipe_emb, recipe_features], dim=1)
        else:
            x = recipe_emb

        # Pass through MLP
        output = self.mlp(x)  # (batch_size, embedding_dim)

        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)

        return output


class TwoTowerModel(nn.Module):
    """
    Two-Tower retrieval model for recommendation.

    Combines user tower and recipe tower with InfoNCE contrastive loss.
    Uses in-batch negative sampling for efficient training.
    """

    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        use_features: bool = False,
        dropout: float = 0.2,
        temperature: float = 0.07,
    ):
        """
        Initialize Two-Tower model.

        Args:
            num_users: Total number of users
            num_recipes: Total number of recipes
            embedding_dim: Embedding dimension for both towers
            hidden_dims: Hidden layer dimensions
            use_features: Whether to use user/recipe features
            dropout: Dropout probability
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temperature = temperature

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            use_features=use_features,
            feature_dim=4,  # 4D taste vectors
            dropout=dropout,
        )

        self.recipe_tower = RecipeTower(
            num_recipes=num_recipes,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            use_features=use_features,
            feature_dim=4,  # 4D taste profiles
            dropout=dropout,
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        recipe_idx: torch.Tensor,
        user_features: torch.Tensor | None = None,
        recipe_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both towers.

        Args:
            user_idx: User indices, shape (batch_size,)
            recipe_idx: Recipe indices, shape (batch_size,)
            user_features: Optional user features, shape (batch_size, 4)
            recipe_features: Optional recipe features, shape (batch_size, 4)

        Returns:
            Tuple of (user_embeddings, recipe_embeddings)
            Both have shape (batch_size, embedding_dim)
        """
        user_emb = self.user_tower(user_idx, user_features)
        recipe_emb = self.recipe_tower(recipe_idx, recipe_features)
        return user_emb, recipe_emb

    def compute_similarity(self, user_emb: torch.Tensor, recipe_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between users and recipes.

        Args:
            user_emb: User embeddings, shape (batch_size, embedding_dim)
            recipe_emb: Recipe embeddings, shape (batch_size, embedding_dim)

        Returns:
            Similarity matrix, shape (batch_size, batch_size)
            Entry (i, j) = similarity between user i and recipe j
        """
        # Compute dot product (since embeddings are normalized, this is cosine similarity)
        similarity = torch.matmul(user_emb, recipe_emb.T) / self.temperature
        return similarity

    def get_user_embeddings(self, user_idx: torch.Tensor, user_features: torch.Tensor | None = None) -> torch.Tensor:
        """Get embeddings for users only."""
        return self.user_tower(user_idx, user_features)

    def get_recipe_embeddings(
        self, recipe_idx: torch.Tensor, recipe_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get embeddings for recipes only."""
        return self.recipe_tower(recipe_idx, recipe_features)


class ColdStartEncoder(nn.Module):
    """Maps user profile features to the shared embedding space used by the Two-Tower model.

    Minimal, robust encoder intended for cold-start users. By default it expects a 4-d taste
    vector but can accept larger feature vectors (equipment, counts, etc.). Output is L2-normalized
    to match the retrieval embedding space.
    """

    def __init__(
        self, feature_dim: int = 4, embedding_dim: int = 64, hidden_dims: list[int] | None = None, dropout: float = 0.2
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        prev = feature_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h

        layers.append(nn.Linear(prev, embedding_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch_size, feature_dim) -> (batch_size, embedding_dim) normalized"""
        out = self.mlp(features)
        return F.normalize(out, p=2, dim=1)


class RetrievalRecommenderModel:
    """Internal wrapper for retrieval-only inference."""

    def __init__(
        self,
        model: TwoTowerModel,
        recipe_embeddings: np.ndarray,
        user_to_idx: dict[str, int],
        recipe_to_idx: dict[str, int],
        idx_to_recipe: dict[int, str],
        users_df: pd.DataFrame | None,
        device: str,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.recipe_embeddings = torch.from_numpy(recipe_embeddings).float().to(device)
        self.user_to_idx = user_to_idx
        self.recipe_to_idx = recipe_to_idx
        self.idx_to_recipe = idx_to_recipe
        self.users_df = users_df
        self.device = device
        self.use_features = model.user_tower.use_features
        self.cold_start_encoder: nn.Module | None = None

    def _get_equipment_compatible_recipes(self, user_id: str, recipes_df: pd.DataFrame | None = None) -> set[int]:
        """Get set of recipe indices compatible with user's owned equipment.

        Args:
            user_id: User identifier
            recipes_df: Optional recipes dataframe for equipment lookup

        Returns:
            Set of recipe indices (idx_to_recipe keys) that have required_equipment
            as subset of user's owned_equipment. If no equipment constraint, returns all.
        """
        if self.users_df is None:
            return set(range(len(self.idx_to_recipe)))

        user_rows = self.users_df[self.users_df["user_id"] == user_id]
        if user_rows.empty:
            return set(range(len(self.idx_to_recipe)))

        user_equipment = set(user_rows.iloc[0].get("owned_equipment") or [])
        if not user_equipment or recipes_df is None:
            return set(range(len(self.idx_to_recipe)))  # No equipment constraint

        compatible_indices = set()
        for idx, recipe_id in self.idx_to_recipe.items():
            recipe_rows = recipes_df[recipes_df["recipe_id"] == recipe_id]
            if not recipe_rows.empty:
                required_eq = set(recipe_rows.iloc[0].get("required_equipment") or [])
                if required_eq.issubset(user_equipment):
                    compatible_indices.add(idx)
            else:
                compatible_indices.add(idx)  # Unknown recipe - include by default

        return compatible_indices if compatible_indices else set(range(len(self.idx_to_recipe)))

    @torch.no_grad()
    def recommend(
        self,
        user_id: str,
        users_df: pd.DataFrame,
        recipes_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """Generate top-N recipe recommendations for a user.

        Args:
            user_id: Target user identifier
            users_df: Users dataframe (users.csv loaded)
            recipes_df: Recipes dataframe (recipes.csv loaded)
            train_df: Training interactions (interactions_train.csv loaded)
            n: Number of recommendations to return

        Returns:
            List of (recipe_id, score) tuples, sorted by score descending.
            Higher scores indicate stronger recommendations.
        """
        # Support cold-start users via optional `cold_start_encoder`.
        user_emb = None
        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)

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

            user_emb = self.model.get_user_embeddings(user_tensor, user_features)
        else:
            # Cold-start path
            if self.cold_start_encoder is None or self.users_df is None:
                raise ValueError(f"Unknown user_id: {user_id}")

            user_row = self.users_df[self.users_df["user_id"] == user_id]
            if user_row.empty:
                raise ValueError(f"Unknown user_id: {user_id}")
            user_row = user_row.iloc[0]

            # Default cold-start features: 4-d taste vector
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
        compatible = self._get_equipment_compatible_recipes(user_id, recipes_df)
        for idx in range(len(similarities)):
            if idx not in compatible:
                similarities[idx] = -float("inf")

        top_scores, top_indices = torch.topk(similarities, k=min(n, len(similarities)))

        return [
            (self.idx_to_recipe[idx.item()], score.item()) for idx, score in zip(top_indices, top_scores, strict=True)
        ]
