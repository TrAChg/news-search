"""Script to index all processed historical illustrations into ChromaDB."""

import re
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embedder import MultimodalEmbedder  # noqa: E402
from src.models.vector_store import VectorStore  # noqa: E402


def parse_crop_metadata(crop_path: Path) -> dict[str, str | int]:
    """Extract metadata from the crop filename (e.g. bpt6k717093t_p1_crop_0.jpg)."""
    filename = crop_path.name
    # Pattern: {issue_id}_p{page}_crop_{index}.jpg
    match = re.match(r"^(.+)_p(\d+)_crop_(\d+)\.jpg$", filename)
    if match:
        issue_id, page_str, crop_idx = match.groups()
        return {
            "issue_id": issue_id,
            "page": int(page_str),
            "crop_index": int(crop_idx),
            "filename": filename,
            "relative_path": crop_path.as_posix(),
        }
    return {
        "filename": filename,
        "relative_path": crop_path.as_posix(),
    }


def index_all_crops(
    processed_dir: Path = Path("data/processed"),
    chroma_dir: Path = Path("data/chroma"),
) -> int:
    """Read all crop images, compute CLIP embeddings, and store them into ChromaDB."""
    crop_files = sorted(
        [
            p
            for p in processed_dir.glob("*_crop_*.jpg")
            if "pipeline_steps" not in p.parts
        ]
    )

    if not crop_files:
        print(f"No crop images found in {processed_dir}. Run pipeline first!")
        return 0

    print(f"Found {len(crop_files)} illustration crops to index.")

    embedder = MultimodalEmbedder()
    store = VectorStore(persist_dir=chroma_dir)

    indexed_count = 0
    for crop_path in crop_files:
        crop_id = crop_path.stem  # e.g. bpt6k717093t_p1_crop_0

        # Skip if already indexed for idempotency
        if store.get(crop_id) is not None:
            print(f"  [Skip] {crop_id} is already indexed.")
            continue

        print(f"  [Embedding] {crop_id} ...")
        embedding = embedder.embed_image(crop_path)
        metadata = parse_crop_metadata(crop_path)

        store.add_illustration(
            illustration_id=crop_id,
            embedding=embedding,
            metadata=metadata,
        )
        indexed_count += 1

    total_in_db = store.count()
    print(f"\nDone! Indexed {indexed_count} new illustrations.")
    print(f"Total illustrations in ChromaDB: {total_in_db}")

    # Sanity-check search
    print("\nRunning verification search for 'un soldat'...")
    query_vec = embedder.embed_text("un soldat")
    results = store.search(query_vec, top_k=3)
    for i, res in enumerate(results, 1):
        print(
            f"  Top {i}: ID={res['id']} | "
            f"Similarity={res['similarity']} | "
            f"Path={res['metadata'].get('relative_path')}"
        )

    return total_in_db


if __name__ == "__main__":
    index_all_crops()
