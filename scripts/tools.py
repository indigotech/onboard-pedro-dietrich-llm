from langchain_core.tools import tool
from langchain_community.utilities import SearchApiAPIWrapper

@tool
def web_search(search_input: str) -> dict:
    """
        Searches updated informations on the internet using the Google search engine.
        Ideal for questions that requires recent data, such as news, ongoing events or constantly changing topics.
    """

    search = SearchApiAPIWrapper()
    return search.results(search_input)


@tool
def add(a: float, b: float) -> float:
    """Given two float numbers as arguments (`a` and `b`), the values are summed (a + b), and the resulting float number is returned."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Given two float numbers as arguments (`a` and `b`), `b` is subtracted from `a` (a - b), and the resulting float number is returned."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Given two float numbers as arguments (`a` and `b`), the values are multiplied (a * b), and the resulting float number is returned."""
    return a * b
