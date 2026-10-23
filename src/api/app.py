"""FastAPI application factory with lifespan model loading and static assets."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router as api_router
from src.models.embedder import MultimodalEmbedder
from src.models.vector_store import VectorStore

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager: Load models once at startup into memory."""
    print("Initializing MultimodalEmbedder and ChromaDB VectorStore...")
    app.state.embedder = MultimodalEmbedder()
    app.state.vector_store = VectorStore(persist_dir=CHROMA_DIR)
    print(f"Ready! Indexed illustrations in DB: {app.state.vector_store.count()}")
    yield
    print("Shutting down search application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Historical News Multimodal Search",
        description=(
            "Multilingual semantic search engine for "
            "French historical newspaper illustrations"
        ),
        version="0.1.0",
        lifespan=cast(Any, lifespan),
    )

    # Enable CORS for local development flexibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    # Mount processed crops directory to serve image files
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static/crops",
        StaticFiles(directory=str(PROCESSED_DIR)),
        name="crops",
    )

    # Mount UI static directory if it exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static/ui",
        StaticFiles(directory=str(STATIC_DIR)),
        name="ui",
    )

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        """Serve the frontend single-page application."""
        index_file = STATIC_DIR / "index.html"
        return FileResponse(str(index_file))

    return app


app = create_app()
