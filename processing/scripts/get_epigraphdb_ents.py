# import argparse
from typing import Any, List

import environs
import pandas as pd
import requests
from icecream import ic
from loguru import logger

from funcs.utils import find_project_root

PROJ_ROOT = find_project_root()
DATA_DIR = PROJ_ROOT.parent / "data"
assert DATA_DIR.exists()
OUTPUT_DIR = DATA_DIR / "epigraphdb_ents"

CHUNK_SIZE = 5_000


def get_node_info(
    meta_node: str, total_length: int, api_url: str, chunk_size: int = 5_000
) -> pd.DataFrame:
    def by_chunk(
        idx: int, url: str, offset: int, chunk_size: int
    ) -> List[Any]:
        logger.info(f"meta_node {meta_node}, chunk #{idx}")
        params = {"limit": chunk_size, "offset": offset, "full_data": False}
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()["results"]

    if total_length < chunk_size:
        chunk_size = total_length
    endpoint = f"/meta/nodes/{meta_node}/list"
    url = f"{api_url}{endpoint}"
    offset_list = [_ for _ in range(0, total_length, chunk_size)]
    logger.info(
        f"""
    meta_node {meta_node}:
    - total_length {total_length}
    - chunks {len(offset_list)}
    """
    )
    nested_res = [
        by_chunk(idx, url, offset, chunk_size)
        for idx, offset in enumerate(offset_list)
    ]
    logger.info(f"meta_node {meta_node} done.")
    res = [
        {"id": item["id"], "name": item["name"]}
        for sub_res in nested_res
        for item in sub_res
    ]
    res_df = pd.DataFrame(res)
    print(res_df.info())
    return res_df


def main():
    env = environs.Env()
    env.read_env()
    api_url = env("EPIGRAPHDB_API_URL")
    backend_url = env("EPIGRAPHDB_WEB_BACKEND_URL")
    ic(api_url)
    ic(backend_url)
    r = [requests.get(f"{_}/ping") for _ in [api_url, backend_url]]
    [_.raise_for_status for _ in r]
    for _ in r:
        assert _.json()

    # get none code meta node names
    r = requests.get(f"{backend_url}/models/meta-nodes/non-code-name")
    r.raise_for_status()
    meta_nodes = r.json()

    # get count of meta nodes
    r = requests.get(f"{backend_url}/about/metrics")
    r.raise_for_status()
    meta_node_counts = {
        _["node_name"]["name"]: _["count"]
        for _ in r.json()["meta_node"]
        if _["node_name"]["name"] in meta_nodes
    }

    # get data
    for meta_node in meta_nodes:
        logger.info(f"meta node: {meta_node}")
        output_file = OUTPUT_DIR / meta_node / "ents.csv"
        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)
            df = get_node_info(
                meta_node=meta_node,
                total_length=meta_node_counts[meta_node],
                api_url=api_url,
                chunk_size=CHUNK_SIZE,
            )
            df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
