"""API routes for historical news illustration search."""

import time

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import HealthResponse, SearchResponse, SearchResultItem

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return health status and count of indexed illustrations."""
    store = getattr(request.app.state, "vector_store", None)
    total = store.count() if store is not None else 0

    return HealthResponse(
        status="ok",
        total_indexed=total,
        model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    )


def calibrate_similarity(
    raw_similarity: float,
    s_min: float = 0.20,
    s_max: float = 0.30,
) -> float:
    """Calibrate raw CLIP cosine similarity into an intuitive [0.0, 1.0] range.

    Due to the multimodal cone effect, unrelated text and historical images
    have an empirical noise floor around 0.18 - 0.20. Real matches scale 0.22+.
    """
    if raw_similarity <= s_min:
        return 0.0
    if raw_similarity >= s_max:
        return 1.0
    return round((raw_similarity - s_min) / (s_max - s_min), 4)


@router.get("/search", response_model=SearchResponse)
async def search_illustrations(
    request: Request,
    q: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=50, description="Max results to retrieve"),
    min_similarity: float = Query(
        0.20,
        ge=0.0,
        le=1.0,
        description="Minimum raw cosine similarity threshold to include",
    ),
) -> SearchResponse:
    """Perform multilingual semantic search over illustrations with calibration."""
    embedder = getattr(request.app.state, "embedder", None)
    store = getattr(request.app.state, "vector_store", None)

    if embedder is None or store is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model or vector store is not initialized.",
        )

    start_time = time.perf_counter()

    # 1. Embed query text into 512D shared multimodal latent space
    query_vector = embedder.embed_text(q)

    # 2. Query HNSW cosine index in ChromaDB
    raw_results = store.search(query_embedding=query_vector, top_k=top_k)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    items: list[SearchResultItem] = []
    for res in raw_results:
        raw_sim = res["similarity"]
        if raw_sim < min_similarity:
            continue

        calibrated = calibrate_similarity(raw_sim)
        is_confident = calibrated >= 0.20

        meta = res.get("metadata", {})
        filename = meta.get("filename", f"{res['id']}.jpg")
        image_url = f"/static/crops/{filename}"

        items.append(
            SearchResultItem(
                id=res["id"],
                similarity=round(raw_sim, 4),
                calibrated_score=round(calibrated, 4),
                is_confident=is_confident,
                distance=round(res["distance"], 4),
                image_url=image_url,
                metadata=meta,
            )
        )

    return SearchResponse(
        query=q,
        count=len(items),
        latency_ms=round(elapsed_ms, 2),
        results=items,
    )
