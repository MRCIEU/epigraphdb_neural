from typing import Any, Dict, List, Optional

import numpy as np
import spacy
import textacy
from fastapi import APIRouter, HTTPException
from scipy.spatial import distance
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
    def _spacy(text, asis, model):
        if asis:
            clean_text = text
        else:
            clean_text = text_processing.preprocess_encode(
                text=text, nlp_model=nlp
            )
        doc = model(clean_text)
        vector = doc.vector.tolist()
        res: models.GetEncodeDict = {
            "clean_text": clean_text,
            "results": vector,
        }
        return res

    def _biosentvec(text, model):
        vector = model.embed_sentence(text).reshape(-1).tolist()
        res: models.GetEncodeDict = {"clean_text": text, "results": vector}
        return res

    nlp = nlp_models.nlp_models[nlp_model.value]
    model_name = nlp_model.value
    model = nlp_models.nlp_models[model_name]
    if model_name in nlp_models.spacy_models:
        return _spacy(text=text, asis=asis, model=model)
    elif model_name in nlp_models.biosentvec_models:
        return _biosentvec(text=text, model=model)
    else:
        raise HTTPException(status_code=422, detail="model not supported")


@router.post("/nlp/encode", response_model=models.PostEncodeResponse)
def post_encode(
    input: models.PostEncodeInput,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
) -> models.PostEncodeDict:
    def _spacy(text_list, asis, model):
        if asis:
            clean_text_list = text_list
        else:
            clean_text_list = [
                text_processing.preprocess_encode(text=_, nlp_model=model)
                for _ in text_list
            ]
        docs = [model(_) for _ in clean_text_list]
        vector_list = [_.vector.tolist() for _ in docs]
        res: models.PostEncodeDict = {
            "clean_text": clean_text_list,
            "results": vector_list,
        }
        return res

    def _biosentvec(text_list, model):
        vector_list = [
            model.embed_sentence(_).reshape(-1).tolist() for _ in text_list
        ]
        res: models.PostEncodeDict = {
            "clean_text": text_list,
            "results": vector_list,
        }
        return res

    text_list = input.text_list
    model_name = nlp_model.value
    model = nlp_models.nlp_models[model_name]
    asis = input.asis
    if model_name in nlp_models.spacy_models:
        return _spacy(text_list=text_list, asis=asis, model=model)
    elif model_name in nlp_models.biosentvec_models:
        return _biosentvec(text_list=text_list, model=model)
    else:
        raise HTTPException(status_code=422, detail="model not supported")


@router.get("/nlp/similarity", response_model=float)
def get_similarity(
    text1: str,
    text2: str,
    nlp_model: nlp_models.NlpModelsEnum = nlp_models.NlpModelsEnum.default,
    asis: bool = True,
) -> Optional[float]:
    def _spacy(text1, text2, asis, model):
        if asis:
            text1_input = text1
            text2_input = text2
        else:
            text1_input = text_processing.preprocess_encode(
                text=text1, nlp_model=model
            )
            text2_input = text_processing.preprocess_encode(
                text=text2, nlp_model=model
            )
        doc1 = model(text1_input)
        doc2 = model(text2_input)
        res = doc1.similarity(doc2)
        return res

    def _biosentvec(text1, text2, model):
        vec1 = model.embed_sentence(text1)
        vec2 = model.embed_sentence(text2)
        res = 1 - distance.cosine(vec1, vec2)
        if np.isnan(res):
            return None
        return res

    model_name = nlp_model.value
    model = nlp_models.nlp_models[model_name]
    if model_name in nlp_models.spacy_models:
        return _spacy(text1=text1, text2=text2, asis=asis, model=model)
    elif model_name in nlp_models.biosentvec_models:
        return _biosentvec(text1=text1, text2=text2, model=model)
    else:
        raise HTTPException(status_code=422, detail="model not supported")


@router.get("/nlp/ner", response_model=List[models.GetNerItem])
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
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        return res

    if nlp_model.value not in nlp_models.ner_models:
        raise HTTPException(status_code=422, detail="model not supported")
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

    if nlp_model.value not in nlp_models.svo_models:
        raise HTTPException(status_code=422, detail="model not supported")
    nlp = nlp_models.nlp_models[nlp_model.value]
    doc = nlp(text)
    svos = list(extract.subject_verb_object_triples(doc))
    res = [_process(_) for _ in svos]
    return res
