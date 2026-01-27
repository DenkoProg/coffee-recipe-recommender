import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss for Two-Tower retrieval models.

    Uses in-batch negatives: For each positive (user, recipe) pair in a batch,
    all other recipes in the batch serve as negatives.

    This is the standard loss used in retrieval models like CLIP, SimCLR, etc.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling logits (default 0.07)
                        Lower values make the model more confident
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb: torch.Tensor, recipe_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss with in-batch negatives.

        Args:
            user_emb: User embeddings, shape (batch_size, embedding_dim)
            recipe_emb: Recipe embeddings, shape (batch_size, embedding_dim)
                       Should be L2-normalized

        Returns:
            Scalar loss value

        The loss treats the batch as follows:
        - Positive pairs: (user[i], recipe[i]) for all i
        - Negative pairs: (user[i], recipe[j]) for all i != j
        """
        batch_size = user_emb.size(0)

        logits = torch.matmul(user_emb, recipe_emb.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class InfoNCELossWithSymmetry(nn.Module):
    """
    Symmetric InfoNCE Loss (bidirectional).

    Computes loss in both directions:
    - User → Recipe classification
    - Recipe → User classification

    This can improve training stability and performance.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize symmetric InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb: torch.Tensor, recipe_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional InfoNCE loss.

        Args:
            user_emb: User embeddings, shape (batch_size, embedding_dim)
            recipe_emb: Recipe embeddings, shape (batch_size, embedding_dim)

        Returns:
            Scalar loss value (average of both directions)
        """
        batch_size = user_emb.size(0)

        logits = torch.matmul(user_emb, recipe_emb.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)

        loss_u2r = F.cross_entropy(logits, labels)
        loss_r2u = F.cross_entropy(logits.T, labels)
        loss = (loss_u2r + loss_r2u) / 2.0

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for retrieval models (alternative to InfoNCE).

    For each anchor (user), compares a positive recipe and a negative recipe.
    Less commonly used than InfoNCE but can be helpful for experimentation.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(self, user_emb: torch.Tensor, positive_emb: torch.Tensor, negative_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            user_emb: Anchor (user) embeddings, shape (batch_size, embedding_dim)
            positive_emb: Positive (recipe) embeddings, shape (batch_size, embedding_dim)
            negative_emb: Negative (recipe) embeddings, shape (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        pos_dist = torch.sum((user_emb - positive_emb) ** 2, dim=1)
        neg_dist = torch.sum((user_emb - negative_emb) ** 2, dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()
