"""ChromaDB-based vector store for recipe embeddings."""

from pathlib import Path
from typing import Any

import numpy as np


try:
    from chromadb import PersistentClient
    from chromadb.errors import NotFoundError
except ImportError:
    PersistentClient = None  # type: ignore
    NotFoundError = Exception  # type: ignore


class VectorStore:
    """
    ChromaDB-based vector store for recipe embeddings.

    Stores:
    - Recipe embeddings with their IDs
    - Optional metadata for each embedding
    - Supports similarity search for candidate retrieval
    """

    DEFAULT_COLLECTION_NAME = "recipe_embeddings"
    DEFAULT_DISTANCE = "cosine"

    def __init__(self, persist_dir: str | Path, collection_name: str | None = None):
        """
        Initialize vector store.

        Args:
            persist_dir: Path to ChromaDB persistence directory
            collection_name: Name of the collection (default: "recipe_embeddings")
        """
        if PersistentClient is None:
            raise ImportError("chromadb is required for VectorStore. Install with: pip install chromadb")

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self._client = None
        self._collection = None

    def _get_client(self) -> Any:
        """Get or create ChromaDB client."""
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = PersistentClient(path=str(self.persist_dir))
        return self._client

    def _get_collection(self, create: bool = False) -> Any:
        """Get collection, optionally creating it."""
        if self._collection is None:
            client = self._get_client()
            if create:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.DEFAULT_DISTANCE},
                )
            else:
                self._collection = client.get_collection(self.collection_name)
        return self._collection

    def exists(self) -> bool:
        """Check if vector store exists and has the collection."""
        if not self.persist_dir.exists():
            return False
        try:
            client = self._get_client()
            client.get_collection(self.collection_name)
            return True
        except (ValueError, NotFoundError):
            return False

    def save(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        reset: bool = False,
        batch_size: int = 256,
    ) -> None:
        """
        Save embeddings to ChromaDB.

        Args:
            embeddings: Numpy array of embeddings (n_items, embedding_dim)
            ids: List of unique IDs for each embedding
            metadatas: Optional list of metadata dicts for each embedding
            reset: If True, delete and recreate collection
            batch_size: Batch size for upserting
        """
        if len(ids) != embeddings.shape[0]:
            raise ValueError(f"ids length ({len(ids)}) does not match embeddings rows ({embeddings.shape[0]}).")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids length.")

        client = self._get_client()

        if reset:
            try:
                client.delete_collection(self.collection_name)
            except (ValueError, NotFoundError):
                pass
            self._collection = None

        collection = self._get_collection(create=True)

        # Upsert in batches
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            batch_ids = ids[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadatas = metadatas[start:end] if metadatas is not None else None

            payload: dict[str, Any] = {
                "ids": batch_ids,
                "embeddings": self._to_list(batch_embeddings),
            }
            if batch_metadatas is not None:
                payload["metadatas"] = batch_metadatas

            collection.upsert(**payload)

        print(f"✅ Vector store saved: {len(ids)} embeddings to {self.persist_dir}")

    def load_embeddings(self, limit: int | None = None) -> tuple[np.ndarray, list[str], list[dict] | None]:
        """
        Load all embeddings from the collection.

        Args:
            limit: Optional limit on number of embeddings to load

        Returns:
            Tuple of (embeddings array, ids list, metadatas list or None)
        """
        collection = self._get_collection()

        get_kwargs: dict[str, Any] = {"include": ["embeddings", "metadatas"]}
        if limit is not None:
            get_kwargs["limit"] = limit

        payload = collection.get(**get_kwargs)
        embeddings = np.array(payload["embeddings"], dtype=np.float32)
        ids = payload["ids"]
        metadatas = payload.get("metadatas")

        print(f"✅ Vector store loaded: {len(ids)} embeddings from {self.persist_dir}")
        return embeddings, ids, metadatas

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: dict | None = None,
    ) -> tuple[list[str], list[float]]:
        """
        Query for similar embeddings.

        Args:
            query_embedding: Query embedding (1D or 2D array)
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            Tuple of (ids list, distances list)
        """
        collection = self._get_collection()

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": self._to_list(query_embedding),
            "n_results": n_results,
        }
        if where is not None:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        # Results are nested lists (one per query)
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results.get("distances") else []

        return ids, distances

    def get_embedding(self, embedding_id: str) -> np.ndarray | None:
        """
        Get a single embedding by ID.

        Args:
            embedding_id: ID of the embedding

        Returns:
            Embedding array or None if not found
        """
        collection = self._get_collection()
        result = collection.get(ids=[embedding_id], include=["embeddings"])

        if result["embeddings"]:
            return np.array(result["embeddings"][0], dtype=np.float32)
        return None

    def count(self) -> int:
        """Get number of embeddings in the collection."""
        collection = self._get_collection()
        return collection.count()

    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by IDs."""
        collection = self._get_collection()
        collection.delete(ids=ids)

    @staticmethod
    def _to_list(embeddings: np.ndarray) -> list[list[float]]:
        """Convert numpy array to list of lists."""
        return embeddings.astype(float).tolist()
