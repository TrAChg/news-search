from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import chromadb


class VectorStore:
    """Persistent vector store using ChromaDB."""

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma",
        collection_name: str = "historical_illustrations",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_illustration(
        self,
        illustration_id: str,
        embedding: Sequence[float],
        metadata: Mapping[str, Any],
    ) -> None:
        """Add a single illustration vector and its metadata to the collection."""
        self.collection.add(
            ids=[illustration_id],
            embeddings=cast(Any, [embedding]),
            metadatas=cast(Any, [metadata]),
        )

    def add_illustrations_batch(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Add multiple illustrations in a single batch operation."""
        self.collection.add(
            ids=list(ids),
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
        )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the top_k most similar illustrations to the query embedding."""
        total_items = self.count()
        if total_items == 0:
            return []

        n_results = min(top_k, total_items)

        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=n_results,
        )

        formatted_results: list[dict[str, Any]] = []

        raw_ids = results.get("ids")
        ids = raw_ids[0] if raw_ids is not None and len(raw_ids) > 0 else []

        raw_distances = results.get("distances")
        distances = (
            raw_distances[0]
            if raw_distances is not None and len(raw_distances) > 0
            else []
        )

        raw_metadatas = results.get("metadatas")
        metadatas = (
            raw_metadatas[0]
            if raw_metadatas is not None and len(raw_metadatas) > 0
            else []
        )

        for i in range(len(ids)):
            item_id = ids[i]
            dist = float(distances[i]) if i < len(distances) else 0.0
            meta = metadatas[i] if i < len(metadatas) else {}

            formatted_results.append(
                {
                    "id": item_id,
                    "distance": dist,
                    "similarity": round(1.0 - dist, 4),
                    "metadata": meta,
                }
            )

        return formatted_results

    def get(self, illustration_id: str) -> dict[str, Any] | None:
        """Retrieve a specific illustration by its ID."""
        result = self.collection.get(ids=[illustration_id])
        raw_ids = result.get("ids")
        if not raw_ids:
            return None

        raw_metas = result.get("metadatas")
        meta = raw_metas[0] if raw_metas is not None and len(raw_metas) > 0 else {}
        return {"id": raw_ids[0], "metadata": meta}

    def delete(self, illustration_id: str) -> None:
        """Delete an illustration from the vector store."""
        self.collection.delete(ids=[illustration_id])

    def count(self) -> int:
        """Return the total number of indexed vectors in the collection."""
        return int(self.collection.count())
