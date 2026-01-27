import argparse
from pathlib import Path

from coffee_recipe_recommender.db.chroma_store import init_collection, upsert_embeddings
from coffee_recipe_recommender.preprocessing.embeddings_io import load_embedding_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a Chroma collection with precomputed embeddings.")
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        required=True,
        help="Path to .npy embeddings file.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/recipes.csv"),
        help="Recipes CSV (must include recipe_id; description is used as document if present).",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/chroma"),
        help="Directory where Chroma persists data.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="recipes",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--distance",
        type=str,
        choices=["cosine", "l2", "ip"],
        default="cosine",
        help="Vector distance function for HNSW.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for upsert.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before inserting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_embedding_bundle(
        embeddings_path=args.embeddings_path,
        recipes_csv_path=args.csv_path,
    )

    collection = init_collection(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        distance=args.distance,
        reset=args.reset,
    )

    upsert_embeddings(
        collection,
        ids=bundle.ids,
        embeddings=bundle.embeddings,
        metadatas=bundle.metadatas,
        documents=bundle.documents,
        batch_size=args.batch_size,
    )

    print(f"Loaded {len(bundle.ids)} embeddings into collection '{args.collection_name}' at {args.persist_dir}.")


if __name__ == "__main__":
    main()
