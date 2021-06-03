from typing import List

from fastapi import HTTPException
from pydantic import BaseModel, validator

from app.nlp_models import config


class PostEncodeInput(BaseModel):
    text_list: List[str]

    @validator("text_list", whole=True)
    def text_list_length(cls, v):
        limit = config.NUM_ENCODE_LIMIT
        if len(v) > limit:
            raise HTTPException(
                status_code=400, detail=f"Too many items. Limit: {limit}."
            )
        return v
