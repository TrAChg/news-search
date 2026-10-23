"""Layout analysis and illustration extractor for historical newspapers."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ExtractedIllustration:
    """Represents a cropped illustration with spatial bounding box and metadata."""

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
    """Extracts illustrations and photographs from historical newspaper page scans."""

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
        min_midtone_ratio: float = 0.40,
        max_black_ratio: float = 0.40,
        min_mean_intensity: float = 45.0,
        max_white_ratio: float = 0.65,
        min_std_intensity: float = 18.0,
        max_laplacian_var: float = 9000.0,
        enable_content_filter: bool = True,
    ) -> None:
        """Initialize detector with geometric, morphological, and content constraints.

        Args:
            min_width: Minimum width in pixels for a valid illustration.
            min_height: Minimum height in pixels for a valid illustration.
            min_area_ratio: Minimum ratio of illustration area to total page area.
            max_area_ratio: Maximum ratio of area to page area (discards page borders).
            min_aspect_ratio: Minimum width/height aspect ratio.
            max_aspect_ratio: Maximum width/height aspect ratio.
            padding_ratio: Margin padding around detected bounding box (fraction).
            morph_kernel_size: Kernel dimensions for morphological closing.
            min_midtone_ratio: Minimum fraction of pixels in midtone range [60, 200].
            max_black_ratio: Maximum fraction of solid black pixels (< 40).
            min_mean_intensity: Minimum average grayscale intensity.
            max_white_ratio: Maximum fraction of paper white background (> 220).
            min_std_intensity: Minimum standard deviation of grayscale pixels.
            max_laplacian_var: Maximum Laplacian variance for text edge density.
            enable_content_filter: If True, evaluates pixel distribution of crops.
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.padding_ratio = padding_ratio
        self.morph_kernel_size = morph_kernel_size
        self.min_midtone_ratio = min_midtone_ratio
        self.max_black_ratio = max_black_ratio
        self.min_mean_intensity = min_mean_intensity
        self.max_white_ratio = max_white_ratio
        self.min_std_intensity = min_std_intensity
        self.max_laplacian_var = max_laplacian_var
        self.enable_content_filter = enable_content_filter

    def is_valid_illustration(self, crop: np.ndarray) -> bool:
        """Validate whether a cropped candidate region is a genuine illustration/photo.

        Filters out:
        1. Solid black blobs (paper tears, shadows, margin ink stains).
        2. Blank paper / empty margins.
        3. Text columns (bimodal high-contrast ink + white background).
        """
        if not self.enable_content_filter:
            return True

        if crop.size == 0:
            return False

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        mean_val = float(np.mean(gray))
        black_ratio = float(np.mean(gray < 40))

        # 1. Reject solid black blobs (scraps, shadows, tears, scan margins)
        if mean_val < self.min_mean_intensity or black_ratio > self.max_black_ratio:
            return False

        white_ratio = float(np.mean(gray > 220))
        std_val = float(np.std(gray))

        # 2. Reject blank paper / bleached crops
        if white_ratio > self.max_white_ratio or std_val < self.min_std_intensity:
            return False

        # 3. Reject text columns (bimodal contrast, low midtones)
        midtone_ratio = float(np.mean((gray >= 60) & (gray <= 200)))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Low midtones combined with white background indicates printed text
        if midtone_ratio < self.min_midtone_ratio and white_ratio > 0.25:
            return False

        # Extreme text edge density combined with low midtone presence
        if lap_var > self.max_laplacian_var and midtone_ratio < 0.50:
            return False

        return True

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
        """Detect illustrations on the provided newspaper page."""
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

        # Build ExtractedIllustration with safety padding and content validation
        illustrations: list[ExtractedIllustration] = []
        for cand in accepted:
            pad_x = int(cand.w * self.padding_ratio)
            pad_y = int(cand.h * self.padding_ratio)
            x_min = max(0, cand.x - pad_x)
            y_min = max(0, cand.y - pad_y)
            x_max = min(w, cand.x + cand.w + pad_x)
            y_max = min(h, cand.y + cand.h + pad_y)

            crop = image[y_min:y_max, x_min:x_max]
            if not self.is_valid_illustration(crop):
                continue

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
        """Save extracted illustration crops to disk as JPEG files."""
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
        """Draw bounding boxes and labels on a copy of the original page image."""
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

    def get_intermediate_stages(
        self, image_input: np.ndarray | Path | str
    ) -> dict[str, np.ndarray]:
        """Extract all intermediate visual steps of the detection pipeline.

        Returns a dictionary mapping step names to their corresponding image arrays:
        - '1_grayscale': Monochrome intensity image
        - '2_gaussian_blur': Grain-smoothed image
        - '3_otsu_binary': Inverted binary ink mask
        - '4_morph_closed': Morphologically fused blocks
        - '5_raw_contours': Input image with all detected raw contours
        - '6_final_illustrations': Input image with final accepted illustration boxes
        """
        image = self._load_image(image_input)

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Otsu Binarization
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 4. Morphological Closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Raw Contours
        contours, _ = cv2.findContours(
            closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        base_color = image.copy()
        if len(base_color.shape) == 2:
            base_color = cv2.cvtColor(base_color, cv2.COLOR_GRAY2BGR)
        raw_contours_img = base_color.copy()
        cv2.drawContours(raw_contours_img, contours, -1, (0, 0, 255), 1)

        # 6. Final Filtered Illustrations
        illustrations = self.detect(image)
        final_boxes_img = self.draw_bounding_boxes(image, illustrations)

        return {
            "1_grayscale": gray,
            "2_gaussian_blur": blurred,
            "3_otsu_binary": thresh,
            "4_morph_closed": closed,
            "5_raw_contours": raw_contours_img,
            "6_final_illustrations": final_boxes_img,
        }

    def save_intermediate_stages(
        self,
        image_input: np.ndarray | Path | str,
        output_dir: Path | str,
        prefix: str = "stage",
    ) -> dict[str, Path]:
        """Save all intermediate pipeline steps as image files to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        stages = self.get_intermediate_stages(image_input)
        saved_paths: dict[str, Path] = {}

        for stage_name, stage_img in stages.items():
            file_path = out_path / f"{prefix}_{stage_name}.jpg"
            success = cv2.imwrite(str(file_path), stage_img)
            if not success:
                raise IOError(f"Failed to write stage image to {file_path}")
            saved_paths[stage_name] = file_path

        return saved_paths
