from typing import Any, Dict, List

from elasticsearch import Elasticsearch

from app.es import index_name_to_meta_node
from app.settings import common_prefix

from . import models


def query_vector(
    query_vector: List[float],
    client: Elasticsearch,
    indices: List[str],
    limit: int = 50,
) -> List[Dict[str, Any]]:
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                # +1 to deal with negative results
                # (script score function must not produce negative scores)
                "source": """cosineSimilarity(
                    params.query_vector,
                    doc["vector"]
                ) + 1
                """.replace(
                    "\n", " "
                ),
                "params": {"query_vector": query_vector},
            },
        }
    }
    response = client.search(
        index=indices,
        body={
            "size": limit,
            "query": script_query,
            "_source": {"includes": ["id", "name", "text"]},
        },
    )
    res = [_ for _ in response["hits"]["hits"]]
    return res


def format_query_results(query_results) -> List[models.EntityQueryItem]:
    res: List[models.EntityQueryItem] = [
        {
            "id": _["_source"]["id"],
            "name": _["_source"]["name"],
            "text": _["_source"]["text"],
            "score": _["_score"] - 1,
            "meta_node": index_name_to_meta_node(_["_index"]),
        }
        for _ in query_results
    ]
    return res


def get_embedding_indices(
    client: Elasticsearch,
) -> List[str]:
    mappings = client.indices.get_mapping()
    embedding_indices = [
        _ for _ in mappings.keys() if _.startswith(common_prefix)
    ]
    return embedding_indices
