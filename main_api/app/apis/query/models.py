from enum import Enum
from typing import List, Optional

from pydantic import create_model_from_typeddict
from typing_extensions import TypedDict


class QueryMethodOptions(str, Enum):
    embeddings = "embeddings"
    simple = "simple"


class EntityQueryItem(TypedDict):
    id: str
    name: str
    text: str
    score: float
    meta_node: str


class GetQueryTextDict(TypedDict):
    clean_text: Optional[str]
    method: str
    results: List[EntityQueryItem]


class GetQueryEntDict(TypedDict):
    method: str
    results: List[EntityQueryItem]


GetQueryTextResponse = create_model_from_typeddict(
    GetQueryTextDict  # type: ignore
)

GetQueryEntResponse = create_model_from_typeddict(
    GetQueryEntDict  # type: ignore
)
