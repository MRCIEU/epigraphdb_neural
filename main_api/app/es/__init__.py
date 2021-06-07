from typing import Dict, List

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from loguru import logger
from pydash import py_
from typing_extensions import Literal

from app.settings import es_host, es_port
from app.utils import chunk_process, timeit

es_client = Elasticsearch(
    "http://{host}:{port}".format(host=es_host, port=es_port),
    verify_certs=True,
)


def es_client_connected() -> bool:
    try:
        es_client.ping()
        return True
    except Exception as e:
        logger.error(e)
        return False


def make_index_name(meta_node: str) -> str:
    template = "embeddings-{meta_node}"
    return template.format(meta_node=meta_node.lower())


def init_index(
    index_name: str, dim_size: int, client: Elasticsearch
) -> Literal[True]:
    client.indices.delete(index=index_name, ignore=[404])
    request_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": -1,
            "index.max_result_window": 100_000,
        },
        "mappings": {
            "dynamic": "true",
            "_source": {"enabled": "true"},
            "properties": {
                "id": {"type": "text"},
                "name": {"type": "text"},
                "text": {"type": "text"},
                "vector": {"type": "dense_vector", "dims": dim_size},
            },
        },
    }
    client.indices.create(
        index=index_name, body=request_body, request_timeout=60
    )
    return True


@timeit
def index_data(
    df: pd.DataFrame,
    index_name: str,
    dim_size: int,
    client: Elasticsearch,
    chunk_size: int = 500,
) -> Literal[True]:
    def _process(
        batch: List[Dict], index_name: str, client: Elasticsearch
    ) -> Literal[True]:
        requests = [
            dict(**{"_op_type": "index", "_index": index_name}, **rec)
            for rec in batch
        ]
        bulk(client, requests)
        return True

    batch_recs = py_.chunk(df.to_dict(orient="records"), chunk_size)
    assert init_index(index_name, dim_size, client)
    process_res = [
        chunk_process(
            chunk_idx=idx,
            size=len(batch_recs),
            func=_process,
            kwargs={
                "batch": batch,
                "index_name": index_name,
                "client": client,
            },
        )
        for idx, batch in enumerate(batch_recs)
    ]
    client.indices.refresh(index=index_name, request_timeout=300)
    assert sum(process_res) == len(process_res)
    return True
