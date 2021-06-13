import sqlite3
from pathlib import Path
from typing import Dict, List

import environs
import numpy as np
import pandas as pd
import requests
from icecream import ic
from loguru import logger
from pydash import py_

from funcs.utils import chunk_process, find_project_root, timeit

VECTOR_DIM = 200  # scispacy sci_lg
ENCODE_URL = "{neural_models_api_url}/nlp/encode"
CHUNK_SIZE = 200  # service limit

PROJ_ROOT = find_project_root()
DATA_DIR = PROJ_ROOT.parent / "data"
EPIGRAPHDB_DIR = DATA_DIR / "epigraphdb_ents"
CLEAN_DIR = EPIGRAPHDB_DIR / "clean"
assert EPIGRAPHDB_DIR.exists()
assert CLEAN_DIR.exists()
OUTPUT_FILE = EPIGRAPHDB_DIR / "epigraphdb_ents.db"


def encode(item_list: List[Dict], encode_url: str) -> List[Dict]:
    text_list = [_["clean_text"] for _ in item_list]
    r = requests.post(encode_url, json={"text_list": text_list, "asis": True})
    r.raise_for_status()
    encodings = r.json()["results"]
    res = [
        {
            "id": _["id"],
            "name": _["name"],
            "clean_text": _["clean_text"],
            "vector": encodings[idx],
        }
        for idx, _ in enumerate(item_list)
    ]
    return res


@timeit
def encode_entity(meta_node: str, ents_dir: Path, encode_url: str) -> None:
    logger.info(f"encode ents for meta_node {meta_node}")

    clean_df_path = ents_dir / "clean" / meta_node / "clean.csv"
    empty_df_path = ents_dir / "encodes" / meta_node / "empty.csv"
    results_df_path = ents_dir / "encodes" / meta_node / "encodes.csv.gz"
    results_df_path.parent.mkdir(exist_ok=True, parents=True)
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

    results_df = encodes_df[~encodes_df["empty"]][
        ["id", "name", "clean_text", "vector"]
    ]
    empty_df = encodes_df[encodes_df["empty"]][["id", "name", "clean_text"]]

    results_df.to_csv(results_df_path, index=False, compression="gzip")

    empty_df.to_csv(empty_df_path, index=False)


@timeit
def write_results(meta_node: str, ents_dir: Path, output_file: Path) -> bool:
    logger.info(f"Processing results for meta_node {meta_node}")
    results_df_path = ents_dir / "encodes" / meta_node / "encodes.csv.gz"
    df = pd.read_csv(results_df_path)
    print(df.info())

    with sqlite3.connect(output_file) as conn:
        df.to_sql(name=meta_node, con=conn, if_exists="replace", index=True)
    return True


def main():
    env = environs.Env()
    env.read_env()
    neural_models_api_url = env("EPIGRAPHDB_NEURAL_MODELS_API")
    ic(neural_models_api_url)
    encode_url = ENCODE_URL.format(neural_models_api_url=neural_models_api_url)
    ic(encode_url)

    meta_nodes = [str(_.stem) for _ in CLEAN_DIR.iterdir() if _.is_dir()]
    ic(meta_nodes)

    # stage 1: encode by meta node
    for meta_node in meta_nodes:
        encode_entity(
            meta_node=meta_node,
            ents_dir=EPIGRAPHDB_DIR,
            encode_url=encode_url,
        )

    # stage 2: collect into one df
    if OUTPUT_FILE.exists():
        logger.info(f"{OUTPUT_FILE} exists, skipped.")
        return
    for meta_node in meta_nodes:
        write_results(
            meta_node=meta_node,
            ents_dir=EPIGRAPHDB_DIR,
            output_file=OUTPUT_FILE,
        )


if __name__ == "__main__":
    main()
