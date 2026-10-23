"""Multimodal embedding module using CLIP."""

from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalEmbedder:
    """CLIP - Contrastive Language-Image Pretraining embedding engine."""

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        img_model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.text_model = SentenceTransformer(text_model_name)
        self.img_model = SentenceTransformer(img_model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embedder text to a list[float]"""
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        return cast(list[float], embedding.tolist())

    def embed_image(self, image_input: Path | str | Image.Image) -> list[float]:
        """Embedder img to list[float]"""
        image: Image.Image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input

        if image.mode != "RGB":
            image = image.convert("RGB")

        embedding = self.img_model.encode(image, convert_to_numpy=True)
        return cast(list[float], embedding.tolist())

    @staticmethod
    def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
