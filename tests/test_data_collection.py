"""Unit tests for GallicaClient."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_collection.gallica_client import GallicaClient


@pytest.fixture
def client() -> GallicaClient:
    """Fixture providing a default GallicaClient instance."""
    return GallicaClient()


# Test 1: Testing Pure Logic
def test_build_iiif_url_default(client: GallicaClient) -> None:
    """Verify standard IIIF image URL formatting."""
    ark_id = "ark:/12148/bpt6k7155522"
    url = client.build_iiif_url(ark_id)
    expected = (
        "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k7155522/f1/full/1024,/0/native.jpg"
    )
    assert url == expected


def test_build_iiif_url_strips_extra_slashes(client: GallicaClient) -> None:
    """Verify that leading and trailing slashes in ARK are safely stripped."""
    ark_id = "/ark:/12148/bpt6k7155522/"
    url = client.build_iiif_url(ark_id, page=3, width=800)
    expected = (
        "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k7155522/f3/full/800,/0/native.jpg"
    )
    assert url == expected


# Test 2: Testing XML Parsing
def test_search_newspaper_parses_xml(client: GallicaClient) -> None:
    """Verify SRU response XML parsing extracts title, date, and ARK ID."""
    mock_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <srw:searchRetrieveResponse xmlns:srw="http://www.loc.gov/zing/srw/" xmlns:dc="http://purl.org/dc/elements/1.1/">
        <srw:records>
            <srw:record>
                <srw:recordData>
                    <dc:title>Le Miroir : entierement illustre</dc:title>
                    <dc:date>1914-08-09</dc:date>
                    <dc:identifier>https://gallica.bnf.fr/ark:/12148/bpt6k7155522</dc:identifier>
                </srw:recordData>
            </srw:record>
        </srw:records>
    </srw:searchRetrieveResponse>
    """

    mock_response = MagicMock()
    mock_response.content = mock_xml
    mock_response.status_code = 200

    with patch(
        "src.data_collection.gallica_client.requests.get", return_value=mock_response
    ):
        results = client.search_newspaper("Le Miroir", 1914, max_records=1)

    assert len(results) == 1
    assert results[0]["title"] == "Le Miroir : entierement illustre"
    assert results[0]["date"] == "1914-08-09"
    assert results[0]["ark_id"] == "ark:/12148/bpt6k7155522"


# Test 3: Testing File Download
def test_download_page_img(client: GallicaClient, tmp_path: Path) -> None:
    """Verify image downloading streams bytes and writes to disk."""
    fake_img_bytes = [b"fake_jpeg_header_", b"fake_pixel_data"]

    mock_response = MagicMock()
    mock_response.iter_content.return_value = fake_img_bytes
    mock_response.status_code = 200

    target_file = tmp_path / "raw" / "test_page.jpg"

    with patch(
        "src.data_collection.gallica_client.requests.get", return_value=mock_response
    ):
        out = client.download_page_img("http://fake-iiif-url.jpg", target_file)

    assert out.exists()
    assert out.read_bytes() == b"fake_jpeg_header_fake_pixel_data"
