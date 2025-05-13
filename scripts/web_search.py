from typing_extensions import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_community.utilities import SearchApiAPIWrapper

class WebSearchInput(TypedDict):
    """Web search input."""
    search_input: Annotated[str, ..., "Search string."]

@tool
def web_search(search_input: str) -> dict:
    """Performs a web search for the given query."""

    search = SearchApiAPIWrapper()
    return search.results(search_input)
