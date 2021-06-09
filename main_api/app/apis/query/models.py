from typing_extensions import TypedDict


class EntityQueryItem(TypedDict):
    id: str
    name: str
    text: str
    score: float
    meta_node: str
