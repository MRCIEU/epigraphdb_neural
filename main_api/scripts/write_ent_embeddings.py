import json
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd
from elasticsearch import Elasticsearch
from icecream import ic
from loguru import logger
from pydash import py_

from app import es
from app.settings import common_prefix
from app.utils import timeit

DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "epigraphdb_ents" / "epigraphdb_ents.db"
CHUNK_SIZE = 500
DIM_SIZE = 200  # model spec


def get_meta_nodes(db_path: Path) -> List[str]:
    with sqlite3.connect(db_path) as conn:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        cursor = conn.cursor()
        cursor.execute(query)
        tables = cursor.fetchall()
        cursor.close()
    ic(tables)
    meta_nodes = py_.flatten_deep(tables)
    ic(meta_nodes)
    return meta_nodes


def purge_existing_indices(client: Elasticsearch):
    es_indices = list(client.indices.get_alias("*").keys())  # type: ignore
    embedding_indices = [_ for _ in es_indices if _.startswith(common_prefix)]
    for index in embedding_indices:
        client.indices.delete(index=index, ignore=[404])


@timeit
def load_node_data(meta_node: str, db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT id, name, clean_text, vector FROM {meta_node}"
        df = pd.read_sql(query, conn)
    print(df.info)
    df = df.assign(
        # convect str vector back to List[float]
        vector=lambda df: df["vector"].apply(lambda x: json.loads(x)),
        text=lambda df: df["clean_text"],
    )
    return df


def main():
    if not DB_PATH.exists():
        logger.info(f"expected file {DB_PATH}. exit now.")

    meta_nodes = get_meta_nodes(DB_PATH)
    purge_existing_indices(client=es.es_client)

    for meta_node in meta_nodes:
        logger.info(f"Process for {meta_node}")
        df = load_node_data(meta_node, DB_PATH)
        index_name = es.meta_node_to_index_name(meta_node)
        es.index_data(
            df=df,
            index_name=index_name,
            dim_size=DIM_SIZE,
            client=es.es_client,
            chunk_size=CHUNK_SIZE,
        )


if __name__ == "__main__":
    main()
