"""Batch and single issue ingestion pipeline for 1914 historical newspapers."""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.pipeline import IngestionPipeline  # noqa: E402

# Curated list of high-yield 1914 illustrated periodicals (WWI photographic issues)
CURATED_1914_ARKS = [
    "ark:/12148/bpt6k717093t",  # Le Miroir, Aug 9, 1914 (Cover mobilization)
    "ark:/12148/bpt6k7170946",  # Le Miroir, Aug 16, 1914 (Belgian front)
    "ark:/12148/bpt6k717095m",  # Le Miroir, Aug 23, 1914 (Artillery & trenches)
    "ark:/12148/bpt6k7170960",  # Le Miroir, Aug 30, 1914 (Troop movements)
    "ark:/12148/bpt6k717097d",  # Le Miroir, Sep 6, 1914 (Battle of the Marne)
]


def run_batch_ingestion(
    ark_list: list[str],
    max_pages: int = 4,
    sleep_between_issues: float = 2.0,
) -> None:
    """Run ingestion pipeline over a list of historical 1914 newspaper ARKs."""
    pipeline = IngestionPipeline()
    total_issues = len(ark_list)

    print(f"\n{'=' * 70}")
    print(f"Starting 1914 Batch Ingestion for {total_issues} periodical issues")
    print(f"Max pages per issue: {max_pages}")
    print(f"{'=' * 70}\n")

    for idx, ark in enumerate(ark_list, start=1):
        clean_name = ark.strip("/").split("/")[-1]
        print(f"[{idx}/{total_issues}] Processing issue: {ark} ({clean_name})...")
        try:
            pipeline.process_issue(ark, max_pages=max_pages)
            print(f"[{idx}/{total_issues}] Successfully processed {ark}\n")
        except Exception as exc:
            print(f"[{idx}/{total_issues}] Error processing {ark}: {exc}\n")

        if idx < total_issues:
            time.sleep(sleep_between_issues)

    print(f"{'=' * 70}")
    print("Batch Ingestion Completed.")
    print("Run `python scripts/index_illustrations.py` to index new crops.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest 1914 historical newspapers from BnF Gallica."
    )
    parser.add_argument(
        "--ark",
        type=str,
        default=None,
        help="Single ARK identifier (e.g. ark:/12148/bpt6k717093t)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Max pages to extract per issue (default: 8)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process the curated list of 1914 WWI illustrated issues",
    )

    args = parser.parse_args()

    if args.ark:
        run_batch_ingestion([args.ark], max_pages=args.max_pages)
    elif args.batch:
        run_batch_ingestion(CURATED_1914_ARKS, max_pages=args.max_pages)
    else:
        # Default behavior: process the primary 1914 issue
        run_batch_ingestion(["ark:/12148/bpt6k717093t"], max_pages=args.max_pages)
