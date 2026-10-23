"""Pydantic schemas for search API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """Single illustration search result item."""

    id: str = Field(..., description="Unique illustration identifier")
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    calibrated_score: float = Field(
        default=0.0, description="Calibrated relevance score (0.0 to 1.0)"
    )
    is_confident: bool = Field(
        default=True, description="Whether the result meets confidence criteria"
    )
    distance: float = Field(..., description="Cosine distance from query vector")
    image_url: str = Field(..., description="URL path to access the cropped image")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the illustration"
    )


class SearchResponse(BaseModel):
    """Response returned by the multimodal search endpoint."""

    query: str = Field(..., description="Original text query")
    count: int = Field(..., description="Number of results returned")
    latency_ms: float = Field(..., description="Search latency in milliseconds")
    results: list[SearchResultItem] = Field(
        default_factory=list, description="Ranked list of matching illustrations"
    )


class HealthResponse(BaseModel):
    """System health and status response."""

    status: str = Field(default="ok", description="Service health status")
    total_indexed: int = Field(
        ..., description="Total number of illustrations indexed in ChromaDB"
    )
    model_name: str = Field(
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        description="Multimodal embedding model name",
    )
