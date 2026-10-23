"""Unit tests for Computer Vision Layout Detector."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data_processing.layout_detector import LayoutDetector


def test_blank_page_returns_no_illustrations() -> None:
    """A uniform blank white page should contain no detected illustrations."""
    detector = LayoutDetector()
    blank_page = np.full((800, 600, 3), 255, dtype=np.uint8)

    results = detector.detect(blank_page)
    assert results == []


def test_detects_prominent_illustration_and_rejects_text_and_noise() -> None:
    """Test detection on a page with photo, text lines, and noise."""
    # 1000x800 white canvas
    page = np.full((1000, 800, 3), 255, dtype=np.uint8)

    # 1. Main illustration: 350w x 400h at (100, 150) with midtone photograph texture
    cv2.rectangle(page, (100, 150), (450, 550), (120, 120, 120), -1)

    # 2. Text column lines: thin strips (w=200, h=10)
    for y_offset in range(600, 750, 25):
        cv2.rectangle(page, (100, y_offset), (300, y_offset + 10), (10, 10, 10), -1)

    # 3. Horizontal separator line (w=700, h=2) -> extreme aspect ratio
    cv2.rectangle(page, (50, 100), (750, 102), (0, 0, 0), -1)

    # 4. Tiny noise artifact (15x15)
    cv2.rectangle(page, (600, 300), (615, 315), (0, 0, 0), -1)

    detector = LayoutDetector(
        min_width=150,
        min_height=150,
        min_area_ratio=0.03,
        min_aspect_ratio=0.3,
        max_aspect_ratio=3.0,
    )

    results = detector.detect(page)

    # Exactly 1 illustration should be kept
    assert len(results) == 1
    ill = results[0]
    bx, by, bw, bh = ill.bbox

    # Bounding box should encompass the 350x400 illustration
    assert bx <= 100 and by <= 150
    assert (bx + bw) >= 450 and (by + bh) >= 550
    assert ill.image_crop.shape[0] == bh
    assert ill.image_crop.shape[1] == bw
    assert ill.confidence > 0.8


def test_rejects_solid_black_blob() -> None:
    """Solid black blobs (ink stains, paper tears) must be filtered out."""
    page = np.full((1000, 800, 3), 255, dtype=np.uint8)
    # Solid black rectangle (mean = 10, black ratio = 100%)
    cv2.rectangle(page, (100, 150), (450, 550), (10, 10, 10), -1)

    detector = LayoutDetector(min_width=150, min_height=150, min_area_ratio=0.03)
    results = detector.detect(page)
    assert len(results) == 0


def test_rejects_text_columns() -> None:
    """Dense newspaper text columns with high contrast must be filtered out."""
    page = np.full((1000, 800, 3), 255, dtype=np.uint8)
    # Simulate a dense column of text lines with high edge density
    for y in range(150, 550, 6):
        cv2.line(page, (100, y), (400, y), (0, 0, 0), 2)

    detector = LayoutDetector(min_width=150, min_height=150, min_area_ratio=0.03)
    results = detector.detect(page)
    assert len(results) == 0


def test_invalid_image_path_raises_value_error(tmp_path: Path) -> None:
    """Non-existent image path should raise a clear ValueError."""
    detector = LayoutDetector()
    non_existent = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ValueError, match="Failed to load image from path"):
        detector.detect(non_existent)


def test_grayscale_input_supported() -> None:
    """Detector should accept 2D single-channel grayscale numpy arrays directly."""
    gray_page = np.full((600, 600), 255, dtype=np.uint8)
    cv2.rectangle(gray_page, (100, 100), (350, 350), 120, -1)

    detector = LayoutDetector(min_width=100, min_height=100, min_area_ratio=0.03)
    results = detector.detect(gray_page)

    assert len(results) == 1
    assert results[0].image_crop.ndim == 2


def test_save_illustrations(tmp_path: Path) -> None:
    """save_illustrations should write crops to disk and return valid Paths."""
    page = np.full((800, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (100, 100), (400, 400), (120, 120, 120), -1)

    detector = LayoutDetector(min_width=100, min_height=100, min_area_ratio=0.03)
    illustrations = detector.detect(page)

    out_dir = tmp_path / "crops"
    saved_paths = detector.save_illustrations(
        illustrations, out_dir, prefix="article_1"
    )

    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].suffix == ".jpg"
    assert "article_1_crop_0" in saved_paths[0].name


def test_draw_bounding_boxes() -> None:
    """draw_bounding_boxes should return an annotated image with boxes drawn."""
    page = np.full((800, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (100, 100), (400, 400), (120, 120, 120), -1)

    detector = LayoutDetector(min_width=100, min_height=100, min_area_ratio=0.03)
    illustrations = detector.detect(page)

    annotated = detector.draw_bounding_boxes(page, illustrations)
    assert annotated.shape == page.shape
    # Ensure drawing modified pixels without altering original page array
    assert not np.array_equal(annotated, page)


def test_page_with_full_border_frame_extracts_inner_illustration() -> None:
    """Outer decorative frames (>85% page) are ignored to extract inner crops."""
    page = np.full((1000, 800, 3), 255, dtype=np.uint8)

    # Outer decorative frame encompassing 95% of the page
    cv2.rectangle(page, (15, 15), (785, 985), (0, 0, 0), 4)

    # Actual illustration inside the frame (350x400)
    cv2.rectangle(page, (150, 200), (500, 600), (120, 120, 120), -1)

    detector = LayoutDetector(
        min_width=150,
        min_height=150,
        min_area_ratio=0.03,
        max_area_ratio=0.85,
    )
    illustrations = detector.detect(page)

    assert len(illustrations) == 1
    bx, by, bw, bh = illustrations[0].bbox
    # Illustration coordinates should match the inner illustration
    assert bx <= 150 and by <= 200
    assert (bx + bw) >= 500 and (by + bh) >= 600
    # Outer frame must not be returned
    assert bw * bh < (1000 * 800 * 0.85)


def test_intermediate_stages_extraction_and_properties() -> None:
    """Verify all pipeline stages are extracted with expected dimensions and values."""
    page = np.full((600, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (100, 100), (350, 350), (120, 120, 120), -1)

    detector = LayoutDetector(min_width=100, min_height=100, min_area_ratio=0.03)
    stages = detector.get_intermediate_stages(page)

    expected_keys = [
        "1_grayscale",
        "2_gaussian_blur",
        "3_otsu_binary",
        "4_morph_closed",
        "5_raw_contours",
        "6_final_illustrations",
    ]
    for key in expected_keys:
        assert key in stages
        assert isinstance(stages[key], np.ndarray)

    # 1. Grayscale should be 2D
    assert stages["1_grayscale"].ndim == 2
    assert stages["1_grayscale"].shape == (600, 600)

    # 2. Gaussian blur should have same shape as grayscale
    assert stages["2_gaussian_blur"].shape == (600, 600)

    # 3. Otsu binary must be 0 and 255 only
    unique_vals = set(np.unique(stages["3_otsu_binary"]))
    assert unique_vals.issubset({0, 255})

    # 4. Morphological closed must also be binary
    unique_morph = set(np.unique(stages["4_morph_closed"]))
    assert unique_morph.issubset({0, 255})

    # 5 & 6. Contours and Final illustration drawings should be 3-channel BGR
    assert stages["5_raw_contours"].ndim == 3
    assert stages["6_final_illustrations"].ndim == 3


def test_save_intermediate_stages_exports_image_files(tmp_path: Path) -> None:
    """Verify that save_intermediate_stages writes valid JPEG files for each step."""
    page = np.full((600, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (100, 100), (350, 350), (40, 40, 40), -1)

    detector = LayoutDetector(min_width=100, min_height=100, min_area_ratio=0.03)
    out_dir = tmp_path / "pipeline_debug"
    saved = detector.save_intermediate_stages(page, out_dir, prefix="step")

    assert len(saved) == 6
    for file_path in saved.values():
        assert file_path.exists()
        assert file_path.stat().st_size > 0
        assert file_path.suffix == ".jpg"
