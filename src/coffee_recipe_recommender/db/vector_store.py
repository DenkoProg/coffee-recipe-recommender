from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.errors import NotFoundError
import numpy as np


def init_collection(
    *,
    persist_dir: Path,
    collection_name: str,
    distance: str,
    reset: bool,
) -> Any:
    persist_dir = persist_dir.expanduser()
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(persist_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
        except (ValueError, NotFoundError):
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance},
    )


def upsert_embeddings(
    collection: Any,
    *,
    ids: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict[str, Any]] | None = None,
    documents: list[str] | None = None,
    batch_size: int = 256,
) -> None:
    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"ids length ({len(ids)}) does not match embeddings rows ({embeddings.shape[0]}).")
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("metadatas length must match ids length.")
    if documents is not None and len(documents) != len(ids):
        raise ValueError("documents length must match ids length.")

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_embeddings = embeddings[start:end]
        batch_metadatas = metadatas[start:end] if metadatas is not None else None
        batch_documents = documents[start:end] if documents is not None else None

        payload: dict[str, Any] = {
            "ids": batch_ids,
            "embeddings": _as_list(batch_embeddings),
        }
        if batch_metadatas is not None:
            payload["metadatas"] = batch_metadatas
        if batch_documents is not None:
            payload["documents"] = batch_documents

        collection.upsert(**payload)


def _as_list(embeddings: np.ndarray | Iterable[Iterable[float]]) -> list[list[float]]:
    if isinstance(embeddings, np.ndarray):
        return embeddings.astype(float).tolist()
    return [list(row) for row in embeddings]
