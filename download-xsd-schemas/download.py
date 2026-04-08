from pathlib import Path
import requests
import lxml.html


def download_xsd_schema(url: str, folder: Path):
    """
        Recursively download XSD schemas and save them to a specified folder.
    """

    folder.mkdir(exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()

    html = lxml.html.fromstring(response.content)
    for link in html.xpath("//a/@href"):
        if link.endswith("../"):
            continue
        elif link.endswith(".xsd"):
            xsd_url = url + link
            xsd_response = requests.get(xsd_url)
            xsd_response.raise_for_status()

            with open(folder / link, "wb") as f:
                f.write(xsd_response.content)
        else:
            download_xsd_schema(url + link, folder / link)


if __name__ == "__main__":
    base_url = "https://fewsdocs.deltares.nl/schemas/version1.0/"
    base_dir = Path("Schemas")

    download_xsd_schema(base_url, base_dir)
