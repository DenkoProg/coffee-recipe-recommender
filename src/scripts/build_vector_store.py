"""Build vector store from trained two-tower model embeddings.

This script:
1. Loads recipe embeddings from a trained two-tower model
2. Saves them to ChromaDB for fast similarity search during inference

Usage:
    uv run python src/scripts/build_vector_store.py \
        --embeddings runs/two_tower/best/recipe_embeddings.npy \
        --checkpoint runs/two_tower/best/two_tower_model.pt \
        --output-dir data/chroma
"""

import argparse
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path

import numpy as np
import torch

from coffee_recipe_recommender.db.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector store from embeddings.")

    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to recipe embeddings .npy file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to two-tower model checkpoint (for idx_to_recipe mapping)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chroma"),
        help="Output directory for ChromaDB",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="recipe_embeddings",
        help="Name of ChromaDB collection",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset (delete) existing collection before saving",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("📂 Loading embeddings...")
    embeddings = np.load(args.embeddings)
    print(f"   Shape: {embeddings.shape}")

    print("📂 Loading checkpoint for ID mapping...")
    with torch.serialization.safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    idx_to_recipe = checkpoint["idx_to_recipe"]
    print(f"   Found {len(idx_to_recipe)} recipe IDs")

    # Create ID list in index order
    ids = [idx_to_recipe[i] for i in range(len(idx_to_recipe))]

    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: {len(ids)} recipe IDs but {embeddings.shape[0]} embeddings")

    # Build metadata (optional - add recipe info here if needed)
    metadatas = [{"idx": i, "recipe_id": rid} for i, rid in enumerate(ids)]

    print(f"💾 Saving to vector store at {args.output_dir}...")
    store = VectorStore(args.output_dir, args.collection_name)
    store.save(embeddings, ids, metadatas=metadatas, reset=args.reset)

    # Verify
    count = store.count()
    print(f"✅ Vector store built: {count} embeddings")


if __name__ == "__main__":
    main()
