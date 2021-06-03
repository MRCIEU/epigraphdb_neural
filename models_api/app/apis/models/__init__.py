import time
from typing import Any, List, Dict

import spacy
import textacy
from textacy import extract
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from loguru import logger

from . import config

logger.info("Loading models")
start_time = time.time()
scispacy_model_lg = spacy.load(config.SCISPACY_MODEL_LG)
# scispacy_model_lg
# - len(doc.vector) == 200
finish_time = time.time()
elapsed_time = round(finish_time - start_time, 2)
logger.info(f"Loading models done in {elapsed_time} seconds")

router = APIRouter()

# TODO: keep models api and remove main api
# TODO: /models/xxx to /nlp/xxx
# TODO: add GET /nlp/models
# TODO: move models to a global module
# TODO: black et al


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


@router.get("/models/encode", response_model=List[float])
def get_encode(text: str) -> List[float]:
    doc = scispacy_model_lg(text)
    vector = doc.vector.tolist()
    return vector


@router.post("/models/encode", response_model=List[List[str]])
def post_encode(input: PostEncodeInput) -> List[List[str]]:
    text_list = input.text_list
    docs = [scispacy_model_lg(_) for _ in text_list]
    res = [_.vector.tolist() for _ in docs]
    return res


@router.get("/models/similarity", response_model=float)
def get_similarity(text1: str, text2: str) -> float:
    doc1 = scispacy_model_lg(text1)
    doc2 = scispacy_model_lg(text2)
    res = doc1.similarity(doc2)
    return res


@router.get("/models/ner")
def get_ner(text: str):
    def _process(ent: spacy.tokens.span.Span):
        res = {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start,
            "end": ent.end,
        }
        return res

    doc = scispacy_model_lg(text)
    res = [_process(_) for _ in doc.ents]
    return res


@router.get("/models/svo")
def get_svo(text: str) -> List[Dict[str, Any]]:
    def _process(triple: textacy.extract.triples.SVOTriple):
        subj = [_.text for _ in triple.subject]
        verb = [_.text for _ in triple.verb]
        obj = [_.text for _ in triple.object]
        res = {
            "subject": subj,
            "verb": verb,
            "object": obj,
        }
        return res

    doc = scispacy_model_lg(text)
    svos = list(extract.subject_verb_object_triples(doc))
    res = [_process(_) for _ in svos]
    return res
