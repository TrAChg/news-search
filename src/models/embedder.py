"""Multimodal embedding module using CLIP."""

from pathlib import Path
from typing import cast

from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalEmbedder:
    """CLIP - Contrastive Language-Image Pretraining embedding engine."""

    def __init__(
        self, model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return cast(list[float], embedding.tolist())

    def embed_image(self, image_input: Path | str | Image.Image) -> list[float]:
        image: Image.Image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input

        if image.mode != "RGB":
            image = image.convert("RGB")

        embedding = self.model.encode(image, convert_to_numpy=True)
        return cast(list[float], embedding.tolist())
