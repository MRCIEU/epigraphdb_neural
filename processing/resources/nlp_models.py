import spacy

from funcs.utils import find_project_root, timeit

from loguru import logger

root = find_project_root()

data_dir = root.parent / "data"
assert data_dir.exists()

models_dir = root.parent / "models"
assert models_dir.exists()

scispacy_lg_path = (
    models_dir / "scispacy"
    / "en_core_sci_lg-0.4.0"
    / "en_core_sci_lg"
    / "en_core_sci_lg-0.4.0"
)

@timeit
def load_scispacy_lg() -> spacy.language.Language:
    scispacy_lg: spacy.language.Language = spacy.load(scispacy_lg_path)
    return scispacy_lg
