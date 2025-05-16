from typing_extensions import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_community.utilities import SearchApiAPIWrapper

class WebSearchInput(TypedDict):
    """Web search input."""
    search_input: Annotated[str, ..., "Search string."]

@tool
def web_search(search_input: str) -> dict:
    """
        Searches updated informations on the internet using the Google search engine.
        Ideal for questions that requires recent data, such as news, ongoing events or constantly changing topics.
    """

    search = SearchApiAPIWrapper()
    return search.results(search_input)
