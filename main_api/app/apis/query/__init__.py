from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query

from app.apis.nlp import get_nlp_encode_text
from app.es import es_client, index_name_to_meta_node, meta_node_to_index_name
from app.utils import vector_empty

from . import funcs, models

router = APIRouter()

try:
    embedding_indices = funcs.get_embedding_indices(client=es_client)
except:
    embedding_indices = []


@router.get("/query/text", response_model=models.GetQueryTextResponse)
def get_query_text(
    text: str,
    asis: bool = False,
    include_meta_nodes: List[str] = Query([]),
    limit: int = Query(50, ge=1, le=200),
) -> models.GetQueryTextDict:
    """Return ents that matches the input text via text embeddings.

    - `asis`: If False, apply builtin preprocessing to `text`
    - `include_meta_nodes`: Leave as is to search in all meta entities,
      otherwise limit to the supplied list
    """

    encode_res = get_nlp_encode_text(text=text, asis=asis)
    query_vector = encode_res["results"]
    clean_text = encode_res["clean_text"]
    if vector_empty(query_vector):
        res: models.GetQueryTextDict = {
            "clean_text": clean_text,
            "results": [],
        }
    else:
        indices = get_indices_from_meta_nodes(include_meta_nodes)
        search_res = funcs.query_vector(
            query_vector, client=es_client, indices=indices, limit=limit
        )
        results = funcs.format_query_results(search_res)
        res = {
            "clean_text": clean_text,
            "results": results,
        }
    return res


@router.get("/query/entity", response_model=List[Dict])
def get_query_ent(
    entity_id: str,
    meta_node: str,
    include_meta_nodes: List[str] = Query([]),
    limit: int = Query(50, ge=1, le=200),
) -> List[models.EntityQueryItem]:
    "Return ents that matches the query entity via text embeddings."
    query_vector = get_query_ent_encode(
        entity_id=entity_id, meta_node=meta_node
    )
    indices = get_indices_from_meta_nodes(include_meta_nodes)
    search_res = funcs.query_vector(
        query_vector, client=es_client, indices=indices, limit=limit
    )
    res: List[models.EntityQueryItem] = funcs.format_query_results(search_res)
    return res


@router.get("/query/entity/encode", response_model=List[float])
def get_query_ent_encode(
    entity_id: str,
    meta_node: str,
) -> List[float]:
    "Return the text embeddings of the query entity."

    index = meta_node_to_index_name(meta_node)
    if index not in embedding_indices:
        raise HTTPException(
            status_code=422,
            detail=f"No such meta entity {meta_node}.",
        )

    r = es_client.search(
        index=index,
        body={"size": 1, "query": {"term": {"id": entity_id}}},
    )
    hits = r["hits"]["hits"]
    if len(hits) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No entity found: {meta_node} {entity_id}.",
        )
    query_vector = r["hits"]["hits"][0]["_source"]["vector"]
    return query_vector


@router.get("/query/meta-entity-list", response_model=List[str])
def get_query_meta_entity_list() -> List[str]:
    """Get currently available meta nodes."""
    res = [index_name_to_meta_node(_) for _ in embedding_indices]
    return res


def get_indices_from_meta_nodes(meta_nodes: List[str]) -> List[str]:
    if len(meta_nodes) > 0:
        indices = [meta_node_to_index_name(_) for _ in meta_nodes]
        indices = list(set(embedding_indices).intersection(set(indices)))
    else:
        indices = embedding_indices
    return indices
