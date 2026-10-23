"""Unit tests for VectorStore."""

from pathlib import Path

import pytest

from src.models.vector_store import VectorStore


@pytest.fixture
def temp_vector_store(tmp_path: Path) -> VectorStore:
    """Fixture providing a VectorStore with an isolated temporary directory."""
    db_dir = tmp_path / "chroma_test"
    return VectorStore(persist_dir=db_dir, collection_name="test_illustrations")


def test_add_and_count(temp_vector_store: VectorStore) -> None:
    """Verify illustrations can be added and counted correctly."""
    assert temp_vector_store.count() == 0

    dummy_vector = [0.1] * 512
    metadata = {
        "issue_id": "bpt6k717093t",
        "page": 1,
        "crop_path": "data/processed/bpt6k717093t_p1_crop_0.jpg",
    }

    temp_vector_store.add_illustration(
        illustration_id="crop_0",
        embedding=dummy_vector,
        metadata=metadata,
    )

    assert temp_vector_store.count() == 1


def test_add_batch(temp_vector_store: VectorStore) -> None:
    """Verify adding multiple illustrations in a batch."""
    ids = ["crop_1", "crop_2", "crop_3"]
    embeddings = [
        [0.1] * 512,
        [0.2] * 512,
        [0.3] * 512,
    ]
    metadatas = [
        {"page": 1},
        {"page": 2},
        {"page": 3},
    ]

    temp_vector_store.add_illustrations_batch(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    assert temp_vector_store.count() == 3


def test_search_ranking(temp_vector_store: VectorStore) -> None:
    """Verify cosine distance ranking finds the most similar vector first."""
    vec_a = [1.0] + [0.0] * 511
    vec_b = [0.0] + [1.0] + [0.0] * 510

    temp_vector_store.add_illustration(
        illustration_id="item_a",
        embedding=vec_a,
        metadata={"label": "soldier"},
    )
    temp_vector_store.add_illustration(
        illustration_id="item_b",
        embedding=vec_b,
        metadata={"label": "ship"},
    )

    query_vec = [1.0] + [0.0] * 511
    results = temp_vector_store.search(query_embedding=query_vec, top_k=2)

    assert len(results) == 2
    top_hit = results[0]
    assert top_hit["id"] == "item_a"
    assert top_hit["metadata"]["label"] == "soldier"
    assert top_hit["distance"] == pytest.approx(0.0, abs=1e-4)
    assert top_hit["similarity"] == pytest.approx(1.0, abs=1e-4)

    # Second result is orthogonal (cosine distance ~1.0)
    second_hit = results[1]
    assert second_hit["id"] == "item_b"
    assert second_hit["distance"] == pytest.approx(1.0, abs=1e-4)
    assert second_hit["similarity"] == pytest.approx(0.0, abs=1e-4)


def test_search_empty_store(temp_vector_store: VectorStore) -> None:
    """Verify searching an empty store gracefully returns an empty list."""
    query_vec = [0.1] * 512
    results = temp_vector_store.search(query_embedding=query_vec, top_k=5)
    assert results == []


def test_persistence(tmp_path: Path) -> None:
    """Verify data is persisted to disk and reloadable in a new client instance."""
    db_dir = tmp_path / "persistent_chroma"

    # 1. First session: insert data
    store_1 = VectorStore(persist_dir=db_dir, collection_name="persistent_coll")
    store_1.add_illustration(
        illustration_id="persisted_1",
        embedding=[0.5] * 512,
        metadata={"title": "Le Figaro 1914"},
    )
    assert store_1.count() == 1

    # 2. Second session: reload from the same directory
    store_2 = VectorStore(persist_dir=db_dir, collection_name="persistent_coll")
    assert store_2.count() == 1

    item = store_2.get("persisted_1")
    assert item is not None
    assert item["metadata"]["title"] == "Le Figaro 1914"


def test_get_and_delete(temp_vector_store: VectorStore) -> None:
    """Verify retrieving and deleting an illustration by ID."""
    temp_vector_store.add_illustration(
        illustration_id="to_delete",
        embedding=[0.2] * 512,
        metadata={"tag": "temporary"},
    )
    assert temp_vector_store.count() == 1

    item = temp_vector_store.get("to_delete")
    assert item is not None
    assert item["metadata"]["tag"] == "temporary"

    temp_vector_store.delete("to_delete")
    assert temp_vector_store.count() == 0
    assert temp_vector_store.get("to_delete") is None
