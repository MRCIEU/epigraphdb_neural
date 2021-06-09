from typing import Any, Dict, List

import spacy
import textacy
from fastapi import APIRouter
from textacy import extract

from app import nlp_models
from app.funcs import text_processing

from . import models

router = APIRouter()


@router.get("/nlp/model-info")
def get_model_info():
    model_info = nlp_models.model_info
    return model_info


@router.get("/nlp/encode", response_model=models.GetEncodeResponse)
def get_encode(
    text: str,
    asis: bool = False,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
) -> models.GetEncodeDict:
    nlp = nlp_models.nlp_models[nlp_model.value]
    if asis:
        clean_text = text
    else:
        clean_text = text_processing.preprocess_encode(
            text=text, nlp_model=nlp
        )
    doc = nlp(clean_text)
    vector = doc.vector.tolist()
    res: models.GetEncodeDict = {"clean_text": clean_text, "results": vector}
    return res


@router.post("/nlp/encode", response_model=models.PostEncodeResponse)
def post_encode(
    input: models.PostEncodeInput,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
) -> models.PostEncodeDict:
    text_list = input.text_list
    nlp = nlp_models.nlp_models[nlp_model.value]
    if input.asis:
        clean_text_list = text_list
    else:
        clean_text_list = [
            text_processing.preprocess_encode(text=_, nlp_model=nlp)
            for _ in text_list
        ]
    docs = [nlp(_) for _ in clean_text_list]
    vector_list = [_.vector.tolist() for _ in docs]
    res: models.PostEncodeDict = {
        "clean_text": clean_text_list,
        "results": vector_list,
    }
    return res


@router.get("/nlp/similarity", response_model=float)
def get_similarity(
    text1: str,
    text2: str,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
) -> float:
    nlp = nlp_models.nlp_models[nlp_model.value]
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    res = doc1.similarity(doc2)
    return res


@router.get("/nlp/ner")
def get_ner(
    text: str,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
):
    def _process(ent: spacy.tokens.span.Span):
        res = {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start,
            "end": ent.end,
        }
        return res

    nlp = nlp_models.nlp_models[nlp_model.value]
    doc = nlp(text)
    res = [_process(_) for _ in doc.ents]
    return res


@router.get("/nlp/svo")
def get_svo(
    text: str,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
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

    nlp = nlp_models.nlp_models[nlp_model.value]
    doc = nlp(text)
    svos = list(extract.subject_verb_object_triples(doc))
    res = [_process(_) for _ in svos]
    return res
