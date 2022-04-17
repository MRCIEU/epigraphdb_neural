from pathlib import Path

from loguru import logger

# params
NUM_ENCODE_LIMIT = 200

# dependency files
SCISPACY_DIR = Path("/models/scispacy")
SCISPACY_LG = (
    SCISPACY_DIR
    / "en_core_sci_lg-0.4.0"
    / "en_core_sci_lg"
    / "en_core_sci_lg-0.4.0"
)
BIOSENVEC_PATH = Path("/models/biosentvec") / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
# SCISPACY_NER_BIONLP13CG_MD = (
#     SCISPACY_DIR
#     / "en_ner_bionlp13cg_md-0.4.0"
#     / "en_ner_bionlp13cg_md"
#     / "en_ner_bionlp13cg_md-0.4.0"
# )

# ----------------------
# sanity check
logger.info("Check dependent files.")
dependent_files = [
    SCISPACY_LG,
    BIOSENVEC_PATH,
    # SCISPACY_MODEL_NER_BIONLP13CG_MD
]
for _ in dependent_files:
    logger.info(f"Dependent file: {_}, {_.exists()}")
    if not _.exists():
        raise OSError(2, "No such file or directory", str(_))
