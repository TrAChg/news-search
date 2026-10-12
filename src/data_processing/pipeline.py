"""Definitely the data pipe_line to download all the papers from Gallica."""

from pathlib import Path

from src.data_collection.gallica_client import GallicaClient
from src.data_processing.layout_detector import LayoutDetector


class IngestionPipeline:
    """Download the paper depend on it's ark_id and max pages."""

    def __init__(
        self,
        gallica_client: GallicaClient | None = None,
        layout_detector: LayoutDetector | None = None,
    ) -> None:
        self.gallica_client = gallica_client or GallicaClient()
        self.layout_detector = layout_detector or LayoutDetector()

    def process_issue(self, ark_id: str, max_pages: int = 1) -> None:
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        debug_dir = Path("data/processed/pipeline_steps")

        client = self.gallica_client
        for page in range(1, max_pages + 1):
            iiif_url = client.build_iiif_url(ark_id, page=page, width=1024)
            clean_name = ark_id.strip("/").split("/")[-1]
            page_path = raw_dir / f"{clean_name}_p{page}.jpg"
            if not page_path.exists():
                client.download_page_img(iiif_url, page_path)

            detector = self.layout_detector
            list_illus = detector.detect(page_path)
            for index, illu in enumerate(list_illus):
                print(
                    f"Id: #{index + 1}| "
                    f"Bound box: {illu.bbox}|"
                    f"Cofidence: {illu.confidence}"
                )

            prefix = f"{clean_name}_p{page}"
            detector.save_illustrations(list_illus, processed_dir, prefix=prefix)
            detector.save_intermediate_stages(page_path, debug_dir, prefix=prefix)
        return None
