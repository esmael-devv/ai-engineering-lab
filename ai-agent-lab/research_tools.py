
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper



search_client = DuckDuckGoSearchAPIWrapper(
    max_results=5,
    region="us-en",
    backend="html"
)



@tool
def web_search(query: str) -> str:
    """Search the public web for information on a topic.
    Returns search result titles, snippets, and source URLs.
    Use this for questions requiring external or recent information.
    """
    if not query.strip():
        return "SEARCH_ERROR: Query cannot be empty."

    try:
        results = search_client.results(
            query=query,
            max_results=5
        )

        if not results:
            return "No search results found."

        formatted = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "")
            url = result.get("link", "")

            formatted.append(
                f"{index}. {title}\n"
                f"   URL: {url}\n"
                f"   Snippet: {snippet}"
            )

        return "\n\n".join(formatted)

    except Exception as error:
        return (
            f"SEARCH_ERROR: {type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    print(web_search.invoke({
        "query": "What is Apache Kafka?"
    }))
