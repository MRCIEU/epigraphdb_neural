from typing import List

import requests
from fastapi import APIRouter

from app.settings import MODELS_ENCODE_URL

from . import models

router = APIRouter()


@router.get("/nlp/encode/text", response_model=List[List[float]])
def get_nlp_encode_text(text: str) -> List[float]:
    "Returns the text embedding of the input text"
    r = requests.get(MODELS_ENCODE_URL, params={"text": text})
    r.raise_for_status()
    res = r.json()
    return res


@router.post("/nlp/encode/text", response_model=List[List[float]])
def post_nlp_encode_text(input: models.PostEncodeInput) -> List[List[float]]:
    """
    Returns the text embedding of the input text.
    POST version for batch processing.
    """
    text_list = input.text_list
    r = requests.post(MODELS_ENCODE_URL, json={"text_list": text_list})
    r.raise_for_status()
    res = r.json()
    return res
