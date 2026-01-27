"""Recommendation evaluation metrics."""

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at position k.

    Args:
        relevances: List of relevance scores (higher is better)
        k: Position cutoff

    Returns:
        DCG@k value
    """
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+2)) for i in range(k)
    # We use i+2 because positions are 1-indexed in the formula
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at position k.

    Args:
        relevances: List of relevance scores in predicted order
        k: Position cutoff (e.g., 5 for NDCG@5)

    Returns:
        NDCG@k value in range [0, 1]
    """
    dcg = dcg_at_k(relevances, k)

    # Ideal DCG: sort relevances in descending order
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(recommendations: list[str], ground_truth: set[str], k: int) -> float:
    """
    Compute Hit Rate@k (recall@k).

    Hit rate is 1 if any ground truth item appears in top-k, else 0.

    Args:
        recommendations: List of recommended item IDs (ordered by relevance)
        ground_truth: Set of relevant item IDs
        k: Position cutoff

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if not ground_truth:
        return 0.0

    top_k = set(recommendations[:k])
    return 1.0 if len(top_k & ground_truth) > 0 else 0.0


def precision_at_k(recommendations: list[str], ground_truth: set[str], k: int) -> float:
    """
    Compute Precision@k.

    Args:
        recommendations: List of recommended item IDs
        ground_truth: Set of relevant item IDs
        k: Position cutoff

    Returns:
        Precision@k value in range [0, 1]
    """
    if k == 0 or not ground_truth:
        return 0.0

    top_k = set(recommendations[:k])
    return len(top_k & ground_truth) / k


def recall_at_k(recommendations: list[str], ground_truth: set[str], k: int) -> float:
    """
    Compute Recall@k.

    Args:
        recommendations: List of recommended item IDs
        ground_truth: Set of relevant item IDs
        k: Position cutoff

    Returns:
        Recall@k value in range [0, 1]
    """
    if not ground_truth:
        return 0.0

    top_k = set(recommendations[:k])
    return len(top_k & ground_truth) / len(ground_truth)


def mean_reciprocal_rank(recommendations: list[str], ground_truth: set[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR is the reciprocal of the rank of the first relevant item.
    If no relevant item is found, MRR is 0.

    Args:
        recommendations: List of recommended item IDs (ordered)
        ground_truth: Set of relevant item IDs

    Returns:
        MRR value in range [0, 1]
    """
    if not ground_truth:
        return 0.0

    for rank, item_id in enumerate(recommendations, start=1):
        if item_id in ground_truth:
            return 1.0 / rank

    return 0.0


def coverage(all_recommendations: list[list[str]], catalog_size: int) -> float:
    """
    Compute catalog coverage.

    Coverage is the percentage of items in the catalog that were recommended
    at least once across all users.

    Args:
        all_recommendations: List of recommendation lists (one per user)
        catalog_size: Total number of items in catalog

    Returns:
        Coverage percentage in range [0, 1]
    """
    if catalog_size == 0:
        return 0.0

    # Collect all unique recommended items
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / catalog_size


def average_popularity(
    all_recommendations: list[list[str]],
    item_popularity: dict[str, int],
) -> float:
    """
    Compute average popularity of recommended items.

    Lower values indicate more diverse recommendations (less popularity bias).

    Args:
        all_recommendations: List of recommendation lists
        item_popularity: Dictionary mapping item_id to popularity score

    Returns:
        Average popularity score
    """
    total_popularity = 0.0
    total_items = 0

    for recs in all_recommendations:
        for item_id in recs:
            total_popularity += item_popularity.get(item_id, 0)
            total_items += 1

    if total_items == 0:
        return 0.0

    return total_popularity / total_items


def evaluate_recommendations(
    recommendations_dict: dict[str, list[str]],
    ground_truth_dict: dict[str, set[str]],
    k: int = 5,
    catalog_size: int | None = None,
) -> dict[str, float]:
    """
    Comprehensive evaluation of recommendations.

    Args:
        recommendations_dict: Dictionary mapping user_id to list of recommended item_ids
        ground_truth_dict: Dictionary mapping user_id to set of relevant item_ids
        k: Position cutoff for metrics
        catalog_size: Total number of items (for coverage calculation)

    Returns:
        Dictionary of metric names to values
    """
    ndcg_scores = []
    hit_rates = []
    mrr_scores = []
    precision_scores = []
    recall_scores = []

    for user_id, recommendations in recommendations_dict.items():
        ground_truth = ground_truth_dict.get(user_id, set())

        if not ground_truth:
            continue

        # Create binary relevances (1 if in ground truth, 0 otherwise)
        relevances = [1.0 if item_id in ground_truth else 0.0 for item_id in recommendations[:k]]

        ndcg_scores.append(ndcg_at_k(relevances, k))
        hit_rates.append(hit_rate_at_k(recommendations, ground_truth, k))
        mrr_scores.append(mean_reciprocal_rank(recommendations, ground_truth))
        precision_scores.append(precision_at_k(recommendations, ground_truth, k))
        recall_scores.append(recall_at_k(recommendations, ground_truth, k))

    metrics = {
        f"NDCG@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"HR@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        f"Precision@{k}": float(np.mean(precision_scores)) if precision_scores else 0.0,
        f"Recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }

    # Add coverage if catalog size is provided
    if catalog_size is not None:
        all_recs = list(recommendations_dict.values())
        metrics["Coverage"] = coverage(all_recs, catalog_size)

    return metrics
