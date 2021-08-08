from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

from app.es import es_client, meta_node_to_index_name
from app.settings import text_common_prefix

from . import models


def query_text(
    text: str, client: Elasticsearch, indices: List[str], limit: int = 50
) -> List[Dict[str, Any]]:
    query = {
        "match": {"name": {"query": text, "operator": "and", "fuzziness": 2}}
    }
    response = client.search(
        index=indices,
        body={
            "size": limit,
            "query": query,
            "_source": {"includes": ["id", "name", "meta_node"]},
        },
    )
    res = [_ for _ in response["hits"]["hits"]]
    return res


def format_query_results(query_results) -> List[models.EntityQueryItem]:
    res: List[models.EntityQueryItem] = [
        {
            "id": _["_source"]["id"],
            "name": _["_source"]["name"],
            "text": _["_source"]["name"],
            "score": _["_score"],
            "meta_node": _["_source"]["meta_node"],
        }
        for _ in query_results
    ]
    return res


def get_ent_name(entity_id: str, meta_node: str) -> Optional[str]:
    index = meta_node_to_index_name(meta_node, type="text")
    text_indices = get_text_indices(client=es_client)
    if index not in text_indices:
        return None
    r = es_client.search(
        index=index, body={"size": 1, "query": {"match": {"id": entity_id}}}
    )
    hits = r["hits"]["hits"]
    # NOTE: currently the id field is a text field in the text indices,
    # need to overhaul the indexing logics
    if len(hits) == 0 or hits[0]["_source"]["id"] != entity_id:
        return None
    entity_name = hits[0]["_source"]["name"]
    return entity_name


def get_text_indices(client: Elasticsearch) -> List[str]:
    mappings = client.indices.get_mapping()
    indices = [_ for _ in mappings.keys() if _.startswith(text_common_prefix)]
    return indices
