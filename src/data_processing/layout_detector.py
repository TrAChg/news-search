"""Layout analysis and illustration extractor for historical newspapers."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ExtractedIllustration:
    """Represents a cropped illustration with spatial bounding box and metadata.

    Attributes:
        bbox: (x, y, width, height) in original image coordinates.
        area: Area in pixels (width * height).
        aspect_ratio: Ratio width / height.
        confidence: Fill confidence score [0.0, 1.0].
        image_crop: Extracted image crop as a numpy uint8 array.
    """

    bbox: tuple[int, int, int, int]
    area: int
    aspect_ratio: float
    confidence: float
    image_crop: np.ndarray


@dataclass
class _CandidateBox:
    x: int
    y: int
    w: int
    h: int
    area: int
    aspect_ratio: float
    confidence: float


class LayoutDetector:
    """Extracts illustrations and photographs from historical newspaper page scans.

    Uses classical OpenCV contour analysis, Otsu adaptive thresholding,
    and morphological filtering to separate large illustrations from text blocks.
    """

    def __init__(
        self,
        min_width: int = 150,
        min_height: int = 150,
        min_area_ratio: float = 0.03,
        max_area_ratio: float = 0.85,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 4.0,
        padding_ratio: float = 0.01,
        morph_kernel_size: tuple[int, int] = (7, 7),
    ) -> None:
        """Initialize detector with geometric and morphological constraints.

        Args:
            min_width: Minimum width in pixels for a valid illustration.
            min_height: Minimum height in pixels for a valid illustration.
            min_area_ratio: Minimum ratio of illustration area to total page area.
            max_area_ratio: Maximum ratio of area to page area (discards page borders).
            min_aspect_ratio: Minimum width/height aspect ratio.
            max_aspect_ratio: Maximum width/height aspect ratio.
            padding_ratio: Margin padding around detected bounding box (fraction).
            morph_kernel_size: Kernel dimensions for morphological closing.
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.padding_ratio = padding_ratio
        self.morph_kernel_size = morph_kernel_size

    def _load_image(self, image_input: np.ndarray | Path | str) -> np.ndarray:
        """Ensure input is an RGB/BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Failed to load image from path: {image_input}")
            return img
        return image_input

    def detect(
        self, image_input: np.ndarray | Path | str
    ) -> list[ExtractedIllustration]:
        """Detect illustrations on the provided newspaper page.

        Args:
            image_input: Numpy image array or path to an image file.

        Returns:
            List of detected ExtractedIllustration instances.
        """
        image = self._load_image(image_input)
        h, w = image.shape[:2]
        page_area = h * w

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Gaussian blur to suppress paper grain / halftone pattern
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu binarization (invert: ink/illustration becomes foreground white)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological closing to fuse dense illustration regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours with tree hierarchy
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[_CandidateBox] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            bbox_area = cw * ch
            area_ratio = bbox_area / float(page_area)

            # Skip outer full-page borders or tiny sub-threshold artifacts
            if area_ratio > self.max_area_ratio or area_ratio < self.min_area_ratio:
                continue
            if cw < self.min_width or ch < self.min_height:
                continue

            aspect_ratio = cw / float(ch)
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            contour_area = cv2.contourArea(cnt)
            confidence = (
                min(1.0, contour_area / float(bbox_area)) if bbox_area > 0 else 0.0
            )

            candidates.append(
                _CandidateBox(
                    x=x,
                    y=y,
                    w=cw,
                    h=ch,
                    area=bbox_area,
                    aspect_ratio=aspect_ratio,
                    confidence=confidence,
                )
            )

        # Sort candidates by area descending (largest first)
        candidates.sort(key=lambda c: c.area, reverse=True)

        # Suppress nested / highly overlapping child candidates
        accepted: list[_CandidateBox] = []
        for cand in candidates:
            is_contained = False
            for acc in accepted:
                ix1 = max(cand.x, acc.x)
                iy1 = max(cand.y, acc.y)
                ix2 = min(cand.x + cand.w, acc.x + acc.w)
                iy2 = min(cand.y + cand.h, acc.y + acc.h)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_area / float(cand.area)) > 0.70:
                        is_contained = True
                        break
            if not is_contained:
                accepted.append(cand)

        # Build ExtractedIllustration with safety padding
        illustrations: list[ExtractedIllustration] = []
        for cand in accepted:
            pad_x = int(cand.w * self.padding_ratio)
            pad_y = int(cand.h * self.padding_ratio)
            x_min = max(0, cand.x - pad_x)
            y_min = max(0, cand.y - pad_y)
            x_max = min(w, cand.x + cand.w + pad_x)
            y_max = min(h, cand.y + cand.h + pad_y)

            crop = image[y_min:y_max, x_min:x_max]
            illustrations.append(
                ExtractedIllustration(
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    area=(x_max - x_min) * (y_max - y_min),
                    aspect_ratio=cand.aspect_ratio,
                    confidence=cand.confidence,
                    image_crop=crop,
                )
            )

        return illustrations

    def save_illustrations(
        self,
        illustrations: list[ExtractedIllustration],
        output_dir: Path | str,
        prefix: str = "crop",
    ) -> list[Path]:
        """Save extracted illustration crops to disk as JPEG files.

        Args:
            illustrations: List of detected ExtractedIllustration objects.
            output_dir: Directory where crops will be written.
            prefix: Filename prefix for the saved files.

        Returns:
            List of Path objects to the saved crop files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for i, ill in enumerate(illustrations):
            filename = f"{prefix}_crop_{i}.jpg"
            file_path = out_path / filename
            success = cv2.imwrite(str(file_path), ill.image_crop)
            if not success:
                raise IOError(f"Failed to write illustration image to {file_path}")
            saved_paths.append(file_path)

        return saved_paths

    def draw_bounding_boxes(
        self,
        image_input: np.ndarray | Path | str,
        illustrations: list[ExtractedIllustration],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the original page image.

        Args:
            image_input: Numpy image array or path to image.
            illustrations: Detected ExtractedIllustration instances.
            color: Box color in BGR tuple (default: bright green).
            thickness: Line thickness in pixels.

        Returns:
            Annotated image copy as a numpy array.
        """
        image = self._load_image(image_input)
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for i, ill in enumerate(illustrations):
            x, y, w, h = ill.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            label = f"#{i + 1} ({w}x{h})"
            cv2.putText(
                annotated,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return annotated
