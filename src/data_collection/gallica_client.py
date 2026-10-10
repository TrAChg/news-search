"""Client for querying the BnF Gallica SRU search and IIIF image APIs."""

from pathlib import Path

from bs4 import BeautifulSoup
from curl_cffi import requests


class GallicaClient:
    """Base Architecture for GallicaClient"""

    def __init__(
        self,
        base_sru_url: str = "https://gallica.bnf.fr/SRU",
        base_iiif_url: str = "https://gallica.bnf.fr/iiif",
    ) -> None:
        self.base_sru_url = base_sru_url.rstrip("/")
        self.base_iiif_url = base_iiif_url.rstrip("/")

    def build_iiif_url(self, ark_id: str, page: int = 1, width: int = 1024) -> str:
        """Cleans and formats the IIIF URL"""
        clean_ark_id = ark_id.strip("/")
        return f"{self.base_iiif_url}/{clean_ark_id}/f{page}/full/{width},/0/native.jpg"

    def search_newspaper(
        self, title: str, year: int, max_records: int = 5
    ) -> list[dict[str, str]]:
        """Queries Gallica SRU and extracts title, publication date, and ARK ID."""
        list_dict_infor = []
        cql_query = (
            f'dc.type all "fascicule" and dc.title all "{title}" and'
            f"gallicapublication_date >= {year} and gallicapublication_date <= {year}"
        )
        params = {
            "operation": "searchRetrieve",
            "version": "1.2",
            "query": cql_query,
            "maximumRecords": max_records,
        }
        response = requests.get(
            self.base_sru_url, params=params, impersonate="chrome120", timeout=15
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        records = soup.find_all("srw:record")
        for record in records:
            dict_infor = {}

            title_tag = record.find("dc:title")
            title = title_tag.get_text(strip=True) if title_tag else "N/A"
            dict_infor["title"] = title

            date_tag = record.find("dc:date")
            date = date_tag.get_text(strip=True) if date_tag else "N/A"
            dict_infor["date"] = date

            ark_id = ""
            for id_tag in record.find_all("dc:identifier"):
                id_text = id_tag.get_text(strip=True)
                if "ark:/" in id_text:
                    ark_id = id_text[id_text.find("ark:/") :]
                    break
            if ark_id:
                dict_infor["ark_id"] = ark_id
            list_dict_infor.append(dict_infor)
        return list_dict_infor

    def download_page_img(self, iiif_url: str, out_path: Path) -> Path:
        """Download img from an IIIF URL and save it to Out Path"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(iiif_url, impersonate="chrome120", timeout=30)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return out_path
