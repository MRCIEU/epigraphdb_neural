from typing import Dict, Union

import requests
from fastapi import APIRouter

from app.settings import MODELS_ENCODE_URL

from . import models

router = APIRouter()


@router.get("/nlp/encode/text", response_model=models.GetNlpEncodeTextResponse)
def get_nlp_encode_text(
    text: str, asis: bool = False
) -> models.GetNlpEncodeTextDict:
    "Returns the text embedding of the input text"
    # there are no preprocessing in main_api yet
    params: Dict[str, Union[str, bool]] = {"text": text, "asis": asis}
    r = requests.get(MODELS_ENCODE_URL, params=params)
    r.raise_for_status()
    res: models.GetNlpEncodeTextDict = r.json()
    return res


@router.post(
    "/nlp/encode/text", response_model=models.PostNlpEncodeTextResponse
)
def post_nlp_encode_text(
    input: models.PostEncodeInput,
) -> models.PostNlpEncodeTextDict:
    """
    Returns the text embedding of the input text.
    POST version for batch processing.
    """
    text_list = input.text_list
    r = requests.post(MODELS_ENCODE_URL, json={"text_list": text_list})
    r.raise_for_status()
    res: models.PostNlpEncodeTextDict = r.json()
    return res
