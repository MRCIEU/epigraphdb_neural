from typing import Any, Dict, List

import spacy
import textacy
from fastapi import APIRouter
from textacy import extract

from app.nlp_models import NlpModelsEnum, nlp_models

from . import models

router = APIRouter()


@router.get("/nlp/models", response_model=List[str])
def get_models_list() -> List[str]:
    available_models = [_.value for _ in NlpModelsEnum]
    return available_models


@router.get("/nlp/encode", response_model=List[float])
def get_encode(
    text: str, nlp_model: NlpModelsEnum = NlpModelsEnum.default
) -> List[float]:
    nlp = nlp_models[nlp_model.value]
    doc = nlp(text)
    vector = doc.vector.tolist()
    return vector


@router.post("/nlp/encode", response_model=List[List[str]])
def post_encode(
    input: models.PostEncodeInput,
    nlp_model: NlpModelsEnum = NlpModelsEnum.default,
) -> List[List[str]]:
    text_list = input.text_list
    nlp = nlp_models[nlp_model.value]
    docs = [nlp(_) for _ in text_list]
    res = [_.vector.tolist() for _ in docs]
    return res


@router.get("/nlp/similarity", response_model=float)
def get_similarity(
    text1: str, text2: str, nlp_model: NlpModelsEnum = NlpModelsEnum.default
) -> float:
    nlp = nlp_models[nlp_model.value]
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    res = doc1.similarity(doc2)
    return res


@router.get("/nlp/ner")
def get_ner(text: str, nlp_model: NlpModelsEnum = NlpModelsEnum.default):
    def _process(ent: spacy.tokens.span.Span):
        res = {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start,
            "end": ent.end,
        }
        return res

    nlp = nlp_models[nlp_model.value]
    doc = nlp(text)
    res = [_process(_) for _ in doc.ents]
    return res


@router.get("/nlp/svo")
def get_svo(
    text: str, nlp_model: NlpModelsEnum = NlpModelsEnum.default
) -> List[Dict[str, Any]]:
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

    nlp = nlp_models[nlp_model.value]
    doc = nlp(text)
    svos = list(extract.subject_verb_object_triples(doc))
    res = [_process(_) for _ in svos]
    return res
