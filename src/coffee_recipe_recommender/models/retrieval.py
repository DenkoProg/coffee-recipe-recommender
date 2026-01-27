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
