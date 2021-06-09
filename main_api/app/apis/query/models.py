from typing import List

from pydantic import create_model_from_typeddict
from typing_extensions import TypedDict


class EntityQueryItem(TypedDict):
    id: str
    name: str
    text: str
    score: float
    meta_node: str


class GetQueryTextDict(TypedDict):
    clean_text: str
    results: List[EntityQueryItem]


GetQueryTextResponse = create_model_from_typeddict(
    GetQueryTextDict  # type: ignore
)
