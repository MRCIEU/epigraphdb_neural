from typing import List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

from app.settings import NUM_ONTOLOGY_DISTANCE_LIMIT, TRANSFORMERS_INFERENCE_URL

router = APIRouter()


class OntologyDistanceInput(BaseModel):
    text_1: List[str]
    text_2: List[str]

    @validator("text_1", each_item=False)
    def text_1_length(cls, v):
        limit = NUM_ONTOLOGY_DISTANCE_LIMIT
        if len(v) > limit:
            raise HTTPException(
                status_code=422, detail=f"Too many items. Limit: {limit}."
            )
        return v

    @validator("text_2", each_item=False)
    def text_2_length(cls, v):
        limit = NUM_ONTOLOGY_DISTANCE_LIMIT
        if len(v) > limit:
            raise HTTPException(
                status_code=422, detail=f"Too many items. Limit: {limit}."
            )
        return v


@router.post("/ontology/distance", response_model=List[float])
def ontology_distance(input_data: OntologyDistanceInput) -> List[float]:
    text_1 = input_data.text_1
    text_2 = input_data.text_2
    r = requests.post(
        TRANSFORMERS_INFERENCE_URL, json={"text_1": text_1, "text_2": text_2}
    )
    r.raise_for_status()
    res: List[float] = r.json()
    return res
