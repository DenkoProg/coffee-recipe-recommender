import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingBundle:
    ids: list[str]
    embeddings: np.ndarray
    metadatas: list[dict[str, Any]] | None
    documents: list[str] | None


def load_embedding_bundle(
    embeddings_path: Path,
    recipes_csv_path: Path,
) -> EmbeddingBundle:
    embeddings = _load_embeddings(embeddings_path)
    ids, metadatas, documents = _load_from_csv(
        recipes_csv_path,
        embeddings.shape[0],
    )
    return EmbeddingBundle(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


def _load_embeddings(embeddings_path: Path) -> np.ndarray:
    embeddings_path = embeddings_path.expanduser()
    if embeddings_path.suffix != ".npy":
        raise ValueError("Embeddings file must be a .npy file.")
    return np.load(embeddings_path, allow_pickle=False)


def _load_from_csv(
    csv_path: Path,
    expected_rows: int,
) -> tuple[list[str], list[dict[str, Any]] | None, list[str] | None]:
    csv_path = csv_path.expanduser()
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("csv_path must be a .csv file.")

    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    documents: list[str] = []
    has_description = False

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "recipe_id" not in fieldnames:
            raise ValueError("Column 'recipe_id' not found in CSV.")
        has_description = "description" in fieldnames
        for row in reader:
            item_id = row.get("recipe_id")
            if not item_id:
                continue
            ids.append(item_id)
            metadata = {key: value for key, value in row.items() if key != "recipe_id" and value not in (None, "")}
            metadatas.append(metadata)
            if has_description:
                documents.append(row.get("description", "") or "")

    if len(ids) != expected_rows:
        raise ValueError(f"CSV rows ({len(ids)}) do not match embeddings rows ({expected_rows}).")

    return ids, metadatas, documents if has_description else None
