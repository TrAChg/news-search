"""Definitely the data pipe_line to download all the papers from Gallica."""

from src.data_processing.pipeline import IngestionPipeline

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.process_issue(
        "ark:/12148/bpt6k717093t"
    )  # example for download and extract the layout from an paper.
