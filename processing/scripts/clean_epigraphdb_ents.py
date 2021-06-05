import re

import pandas as pd
from loguru import logger

from funcs.utils import find_project_root

GWAS_SOURCE_EXCLUDE = [
    "eqtl-a-",  # ENSGXXXXX
    "met-b-",  # e.g. B:%Mature
    "prot-c-",  # gene / prot codes
]
GWAS_PATTERN_EXCLUDE = [
    {"source": "ubm-a-", "pattern": r"NET\d+ \d+"},
    {"source": "met-a-", "pattern": r"X-\d+"},
]

PROJ_ROOT = find_project_root()
DATA_DIR = PROJ_ROOT.parent / "data"
OUTPUT_DIR = DATA_DIR / "epigraphdb_ents"
assert DATA_DIR.exists()
assert OUTPUT_DIR.exists()


def process_gwas(df: pd.DataFrame) -> pd.DataFrame:
    def _exclude_pat(
        row_data: pd.Series, exclude_source: str, exclude_prog: re.Pattern
    ) -> bool:
        if not row_data["id"].startswith(exclude_source):
            return True
        elif exclude_prog.match(row_data["name"]) is None:
            return True
        else:
            return False

    df["name"] = df["name"].astype(str)

    # names that are "NA"
    logger.info("Drop 'NA' items.")
    n = len(df)
    df = df[df["name"] != "NA"]
    logger.info(f"Dropped {len(df) - n} items.")

    # remove exclude source
    logger.info("Drop exclude source items.")
    n = len(df)
    df = df[
        df["id"].apply(
            lambda id: sum([not id.startswith(_) for _ in GWAS_SOURCE_EXCLUDE])
            == len(GWAS_SOURCE_EXCLUDE)
        )
    ].reset_index(drop=True)
    logger.info(f"Dropped {n - len(df):_} items.")

    # UKB: remove common
    # NOTE: common prefix should be part of the encode text in most cases
    # and by eyeballing traits with common prefix are not overwhelmingly
    # prevalent
    # so not doing this for now

    # Remove code names
    logger.info("Drop code name pattern items.")
    for pattern in GWAS_PATTERN_EXCLUDE:
        source = pattern["source"]
        pat = pattern["pattern"]
        n = len(df)
        logger.info(f"source: {source}, pattern: {pat}")
        prog = re.compile(pat)
        df = df[
            df.apply(lambda row: _exclude_pat(row, source, prog), axis=1)
        ].reset_index(drop=True)
        logger.info(f"Dropped {n - len(df):_} items.")
    return df


def main():
    meta_nodes = [str(_.stem) for _ in OUTPUT_DIR.iterdir() if _.is_dir()]
    logger.info(f"meta nodes: {meta_nodes}")

    process_funcs = {"Gwas": process_gwas}

    for meta_node in meta_nodes:
        logger.info(f"Process {meta_node}")
        input_file = OUTPUT_DIR / meta_node / "ents.csv"
        assert input_file.exists()
        output_file = OUTPUT_DIR / meta_node / "clean.csv"
        if not output_file.exists():
            df = pd.read_csv(input_file)
            if meta_node in process_funcs.keys():
                df = process_funcs[meta_node](df)
            df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
