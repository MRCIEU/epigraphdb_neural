from typing import List

import numpy as np
import spacy


def preprocess_encode(text: str, nlp_model: spacy.language.Language) -> str:
    text_raw = text
    text_lowered = lower_titled(text_raw)
    # use the lemmatised tokens as text
    doc = nlp_model(text_lowered)
    clean_text = " ".join([token.lemma_ for token in doc])
    res = pick_valid_text(
        candidates=[clean_text, text_lowered, text_raw], nlp_model=nlp_model
    )
    return res


def lower_titled(text: str) -> str:
    """
    'Body mass index' -> 'body mass index',
    'FEVER' -> 'FEVER'
    """

    def _f(text: str) -> str:
        try:
            if text[0].isupper() and text[1].islower():
                return text[0].lower() + text[1:]
            else:
                return text
        except:
            return text

    res = " ".join(_f(_) for _ in text.split(" "))
    return res


def pick_valid_text(
    candidates: List[str], nlp_model: spacy.language.Language
) -> str:
    """Pick valid text from `candidates` (ordered from most preferred
    to least preferred) the first candidate that is able to produce
    embedded vector. If all failed, pick the first one.
    """
    most_preferred = candidates[0]
    for candidate in candidates:
        doc = nlp_model(candidate)
        if np.count_nonzero(doc.vector) != 0:
            return candidate
    return most_preferred
