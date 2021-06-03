import time
from enum import Enum

import spacy
from loguru import logger

from . import config

logger.info("Loading models")
start_time = time.time()
scispacy_lg = spacy.load(config.SCISPACY_LG)
# scispacy_model_lg
# - len(doc.vector) == 200
finish_time = time.time()
elapsed_time = round(finish_time - start_time, 2)
logger.info(f"Loading models done in {elapsed_time} seconds")

default_model = scispacy_lg

nlp_models = {
    "default": default_model,
    "scispacy_lg": scispacy_lg,
}
NlpModelsEnum = Enum(  # type: ignore
    "NlpModelsEnum", {_: _ for _ in nlp_models.keys()}
)
