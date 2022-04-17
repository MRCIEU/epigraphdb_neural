import time
from enum import Enum
from typing import Dict, Optional, Union, Any

import spacy
from loguru import logger
import sent2vec
from typing_extensions import TypedDict

from . import config

# ==== types ====


class ModelInfoItem(TypedDict):
    desc: str
    vector: bool
    vector_dim: Optional[float]
    ner: bool
    svo: bool


class ModelInfo(TypedDict):
    meta: Dict[str, str]
    models: Dict[str, ModelInfoItem]

NlpModel = Union[spacy.language.Language, sent2vec.Sent2vecModel]

# ==== loading models ====

logger.info("Loading models")
start_time = time.time()
scispacy_lg: spacy.language.Language = spacy.load(config.SCISPACY_LG)
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model(str(config.BIOSENVEC_PATH))
finish_time = time.time()
elapsed_time = round(finish_time - start_time, 2)
logger.info(f"Loading models done in {elapsed_time} seconds")

default_model = scispacy_lg

# ==== rest ====

nlp_models: Dict[str, NlpModel] = {
    "default": default_model,
    "scispacy_lg": scispacy_lg,
}
NlpModelsEnum = Enum(  # type: ignore
    "NlpModelsEnum", {_: _ for _ in nlp_models.keys()}
)

model_info: ModelInfo = {
    "meta": {
        "desc": "str; Model description",
        "vector": "True if model can encode text to vector",
        "vector_dim": "Float if vector else None",
        "ner": "True if model has NER output",
        "svo": "True if model has svo output",
    },
    "models": {
        "scispacy_lg": {
            "desc": """
            en_core_sci_lg-0.4.0.
            A full spaCy pipeline for biomedical data
            with a ~785k vocabulary and 600k word vectors.
            """,
            "vector": True,
            "vector_dim": 200,
            "ner": True,
            "svo": True,
        }
    },
}
model_info["models"]["default"] = model_info["models"]["scispacy_lg"]
