from typing import List

from fastapi import HTTPException
from pydantic import BaseModel, create_model_from_typeddict, validator
from typing_extensions import TypedDict

from app.settings import NUM_ENCODE_LIMIT, NUM_SIM_LIMIT


class PostEncodeInput(BaseModel):
    text_list: List[str]
    asis: bool = False

    @validator("text_list", each_item=False)
    def text_list_length(cls, v):
        limit = NUM_ENCODE_LIMIT
        if len(v) > limit:
            raise HTTPException(
                status_code=422, detail=f"Too many items. Limit: {limit}."
            )
        return v


class SimilarityTextInput(BaseModel):
    text_list: List[str]

    @validator("text_list", each_item=False)
    def text_list_length(cls, v):
        limit = NUM_SIM_LIMIT
        if len(v) > limit:
            raise HTTPException(
                status_code=422, detail=f"Too many items. Limit: {limit}."
            )
        return v


class SimilarityTextResponseItem(BaseModel):
    text_a: str
    text_b: str
    similarity_score: float


SimilarityTextResponse = List[SimilarityTextResponseItem]


class GetNlpEncodeTextDict(TypedDict):
    clean_text: str
    results: List[float]


class PostNlpEncodeTextDict(TypedDict):
    clean_text: List[str]
    results: List[List[float]]


GetNlpEncodeTextResponse = create_model_from_typeddict(
    GetNlpEncodeTextDict  # type: ignore
)
PostNlpEncodeTextResponse = create_model_from_typeddict(
    PostNlpEncodeTextDict  # type: ignore
)
