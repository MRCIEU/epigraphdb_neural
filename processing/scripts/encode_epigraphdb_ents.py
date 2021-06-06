import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import environs
import numpy as np
import pandas as pd
import requests
from icecream import ic
from loguru import logger
from pydash import py_

from funcs.utils import find_project_root

VECTOR_DIM = 200  # scispacy sci_lg
ENCODE_URL = "{neural_api_url}/nlp/encode"
CHUNK_SIZE = 200  # service limit

PROJ_ROOT = find_project_root()
DATA_DIR = PROJ_ROOT.parent / "data"
ENTS_DIR = DATA_DIR / "epigraphdb_ents"
OUTPUT_FILE = ENTS_DIR / "epigraphdb_ents.db"
assert ENTS_DIR.exists()


def chunk_process(
    chunk_idx: int, size: int, func: Callable, kwargs: Dict
) -> Any:
    logger.info(f"#{chunk_idx}/{size}")
    return func(**kwargs)


def encode(item_list: List[Dict], encode_url: str) -> List[Dict]:
    text_list = [_["name"] for _ in item_list]
    r = requests.post(encode_url, json={"text_list": text_list})
    r.raise_for_status()
    encodings = r.json()
    res = [
        {"id": _["id"], "name": _["name"], "vector": encodings[idx]}
        for idx, _ in enumerate(item_list)
    ]
    return res


def encode_entity(meta_node: str, ents_dir: Path, encode_url: str) -> None:
    start_time = time.time()
    logger.info(f"encode ents for meta_node {meta_node}")

    clean_df_path = ents_dir / meta_node / "clean.csv"
    empty_df_path = ents_dir / meta_node / "empty.csv"
    results_df_path = ents_dir / meta_node / "encodes.csv.gz"
    if results_df_path.exists():
        logger.info(f"{results_df_path} exists, skipped")
        return
    assert clean_df_path.exists()
    df = pd.read_csv(clean_df_path)
    print(df.info())
    empty_vector = np.zeros(VECTOR_DIM)

    recs = df.to_dict(orient="records")
    nested_recs = py_.chunk(recs, CHUNK_SIZE)
    encodes_df = pd.DataFrame(
        py_.flatten(
            [
                chunk_process(
                    chunk_idx=idx,
                    size=len(nested_recs),
                    func=encode,
                    kwargs={"item_list": _, "encode_url": encode_url},
                )
                for idx, _ in enumerate(nested_recs)
            ]
        )
    )
    encodes_df = encodes_df.assign(
        empty=lambda df: df["vector"].apply(
            lambda x: np.array_equiv(np.array(x), empty_vector)
        )
    )

    results_df = encodes_df[~encodes_df["empty"]][["id", "name", "vector"]]
    empty_df = encodes_df[encodes_df["empty"]][["id", "name"]]

    results_df.to_csv(results_df_path, index=False, compression="gzip")

    empty_df.to_csv(empty_df_path, index=False)

    finish_time = time.time()
    elasped_mins = round((finish_time - start_time) / 60, 2)
    logger.info(f"encode finished in {elasped_mins} mins")


def write_results(meta_node: str, ents_dir: Path, output_file: Path) -> bool:
    logger.info(f"Processing results for meta_node {meta_node}")
    start_time = time.time()
    results_df_path = ents_dir / meta_node / "encodes.csv.gz"
    df = pd.read_csv(results_df_path)
    print(df.info())

    with sqlite3.connect(output_file) as conn:
        df.to_sql(name=meta_node, con=conn, if_exists="replace", index=True)
    finish_time = time.time()
    elasped_mins = round((finish_time - start_time) / 60, 2)
    logger.info(f"Processing in {elasped_mins} mins")
    return True


def main():
    env = environs.Env()
    env.read_env()
    neural_api_url = env("EPIGRAPHDB_NEURAL_API")
    ic(neural_api_url)
    encode_url = ENCODE_URL.format(neural_api_url=neural_api_url)
    ic(encode_url)

    meta_nodes = [str(_.stem) for _ in ENTS_DIR.iterdir() if _.is_dir()]
    ic(meta_nodes)

    # stage 1: encode by meta node
    for meta_node in meta_nodes:
        encode_entity(
            meta_node=meta_node, ents_dir=ENTS_DIR, encode_url=encode_url,
        )

    # stage 2: collect into one df
    if OUTPUT_FILE.exists():
        logger.info(f"{OUTPUT_FILE} exists, skipped.")
        return
    for meta_node in meta_nodes:
        write_results(
            meta_node=meta_node, ents_dir=ENTS_DIR, output_file=OUTPUT_FILE,
        )


if __name__ == "__main__":
    main()