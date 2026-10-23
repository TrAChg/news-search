"""Unit tests for MultimodalEmbedder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.models.embedder import MultimodalEmbedder


@pytest.fixture
def mock_embedder() -> tuple[MultimodalEmbedder, MagicMock]:
    """Fixture providing a MultimodalEmbedder with mocked SentenceTransformer."""
    with patch("src.models.embedder.SentenceTransformer") as mock_st_cls:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance
        embedder = MultimodalEmbedder("test-model")
        return embedder, mock_instance


def test_embed_text(mock_embedder: tuple[MultimodalEmbedder, MagicMock]) -> None:
    """Verify embed_text encodes text and returns a list of floats."""
    embedder, mock_model = mock_embedder
    fake_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_model.encode.return_value = fake_embedding

    result = embedder.embed_text("sample headline")

    mock_model.encode.assert_called_once_with("sample headline", convert_to_numpy=True)
    assert result == pytest.approx([0.1, 0.2, 0.3])


def test_embed_image_from_pil(
    mock_embedder: tuple[MultimodalEmbedder, MagicMock],
) -> None:
    """Verify embed_image handles direct PIL Image and converts mode if needed."""
    embedder, mock_model = mock_embedder
    fake_embedding = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mock_model.encode.return_value = fake_embedding

    img = Image.new("RGBA", (10, 10), color="red")
    result = embedder.embed_image(img)

    assert result == pytest.approx([0.4, 0.5, 0.6])
    assert mock_model.encode.call_count == 1
    encoded_arg = mock_model.encode.call_args[0][0]
    assert encoded_arg.mode == "RGB"


def test_embed_image_from_path(
    mock_embedder: tuple[MultimodalEmbedder, MagicMock], tmp_path: Path
) -> None:
    """Verify embed_image loads image from file path and encodes it."""
    embedder, mock_model = mock_embedder
    fake_embedding = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    mock_model.encode.return_value = fake_embedding

    image_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(image_path)

    result = embedder.embed_image(image_path)

    assert result == pytest.approx([0.7, 0.8, 0.9])
    assert mock_model.encode.call_count == 1


def test_compar_text_img() -> None:
    """Verify the simularity between the same words but with diff languges."""
    embedder = MultimodalEmbedder()
    img_imput_path = Path("data/processed/bpt6k717093t_p1_crop_0.jpg")

    vec_fr = embedder.embed_text("un soldat")
    vec_en = embedder.embed_text("a soldier")
    vec_vn = embedder.embed_text("một chiến binh")
    vec_img = embedder.embed_image(img_imput_path)

    similarity_fr_en = embedder.compute_similarity(vec_fr, vec_en)
    similarity_vn_en = embedder.compute_similarity(vec_en, vec_vn)
    similarity_img_text = embedder.compute_similarity(vec_en, vec_img)

    assert similarity_img_text >= 0.2
    assert similarity_vn_en >= 0.8
    assert similarity_fr_en >= 0.8
