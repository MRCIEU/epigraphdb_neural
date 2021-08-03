from typing import List

from fastapi import HTTPException
from pydantic import BaseModel, validator
from typing_extensions import TypedDict

from app.nlp_models import config


class PostEncodeInput(BaseModel):
    text_list: List[str]
    asis: bool = False

    @validator("text_list", whole=True)
    def text_list_length(cls, v):
        limit = config.NUM_ENCODE_LIMIT
        if len(v) > limit:
            raise HTTPException(
                status_code=400, detail=f"Too many items. Limit: {limit}."
            )
        return v


class GetEncodeDict(TypedDict):
    clean_text: str
    results: List[float]


class PostEncodeDict(TypedDict):
    clean_text: List[str]
    results: List[List[float]]


# NOTE: can't use pydantic's own conversion
# as this is pinned with spacy's pinned version of older pydantic
class GetEncodeResponse(BaseModel):
    clean_text: str
    results: List[float]


class PostEncodeResponse(BaseModel):
    clean_text: List[str]
    results: List[List[float]]


class GetNerItem(BaseModel):
    text: str
    label: str
    start: int
    end: int
    start_char: int
    end_char: int
