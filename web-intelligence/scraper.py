
import requests
from bs4 import BeautifulSoup


def fetch_website_contents(url: str) -> str:
    if not url.startswith(("https://", "http://")):
        url = "https://" + url

    headers = {
        "User-Agent": "AIEngineeringLab/1.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "text/html" not in content_type.lower():
        raise ValueError("The URL does not return an HTML webpage.")

    soup = BeautifulSoup(response.text, "html.parser")

    title = (
        soup.title.get_text(strip=True)
        if soup.title
        else "Untitled webpage"
    )

    for tag in soup([
        "script", "style", "nav",
        "footer", "header", "noscript"
    ]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError("No readable content found.")

    # Keep input size manageable for the LLM.
    text = text[:12000]

    return f"Title: {title}\n\nContent:\n{text}"


if __name__ == "__main__":
    url = input("Enter website URL: ").strip()

    try:
        print(fetch_website_contents(url))
    except (requests.RequestException, ValueError) as error:
        print(f"Scraping failed: {error}")
