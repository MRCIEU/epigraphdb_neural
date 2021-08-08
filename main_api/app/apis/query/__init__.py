from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.apis.nlp import get_nlp_encode_text
from app.es import es_client, index_name_to_meta_node, meta_node_to_index_name
from app.utils import vector_empty

from . import embeddings_funcs, models, text_funcs

router = APIRouter()


@router.get("/query/text", response_model=models.GetQueryTextResponse)
def get_query_text(
    text: str,
    asis: bool = False,
    include_meta_nodes: List[str] = Query([]),
    method: models.QueryMethodOptions = models.QueryMethodOptions.embeddings,
    limit: int = Query(50, ge=1, le=200),
) -> models.GetQueryTextDict:
    """Return ents that matches the input text via text embeddings.

    - `asis`: If False, apply builtin preprocessing to `text`.
      NOTE: this only applies when `method` is "embeddings"
    - `method`:
      - "embeddings" (default): search for entities based on
        semantic similarities of text embeddings
      - "simple": search for entities based on simple text matching
    - `include_meta_nodes`: Leave as is to search in all meta entities,
      otherwise limit to the supplied list
    """

    if method.value == "embeddings":
        encode_res = get_nlp_encode_text(text=text, asis=asis)
        query_vector = encode_res["results"]
        clean_text = encode_res["clean_text"]
        if vector_empty(query_vector):
            res: models.GetQueryTextDict = {
                "clean_text": clean_text,
                "method": "embeddings",
                "results": [],
            }
        else:
            indices = get_indices_from_meta_nodes(
                include_meta_nodes, type="embeddings"
            )
            search_res = embeddings_funcs.query_vector(
                query_vector, client=es_client, indices=indices, limit=limit
            )
            results = embeddings_funcs.format_query_results(search_res)
            res = {
                "clean_text": clean_text,
                "method": "embedddings",
                "results": results,
            }
    else:
        # NOTE: for endpoint use "simple",
        # for things further use "text""
        indices = get_indices_from_meta_nodes(include_meta_nodes, type="text")
        search_res = text_funcs.query_text(
            text=text, client=es_client, indices=indices, limit=limit
        )
        results = text_funcs.format_query_results(search_res)
        res = {
            "clean_text": None,
            "method": "embeddings",
            "results": results,
        }
    return res


@router.get("/query/entity", response_model=models.GetQueryEntResponse)
def get_query_ent(
    entity_id: str,
    meta_node: str,
    include_meta_nodes: List[str] = Query([]),
    method: models.QueryMethodOptions = models.QueryMethodOptions.embeddings,
    limit: int = Query(50, ge=1, le=200),
) -> models.GetQueryEntDict:
    """Return ents that matches the query entity via text embeddings.

    - `method`:
      - "embeddings" (default): search for entities based on
        semantic similarities of text embeddings
      - "simple": search for entities based on simple text matching
    """
    if method.value == "embeddings":
        try:
            query_vector = get_query_ent_encode(
                entity_id=entity_id, meta_node=meta_node
            )
        except:
            # When the query term is not encoded
            except_res: models.GetQueryEntDict = {
                "method": "embeddings",
                "results": [],
            }
            return except_res
        indices = get_indices_from_meta_nodes(
            include_meta_nodes, type=method.value
        )
        search_res = embeddings_funcs.query_vector(
            query_vector, client=es_client, indices=indices, limit=limit
        )
        results: List[
            models.EntityQueryItem
        ] = embeddings_funcs.format_query_results(search_res)
        res: models.GetQueryEntDict = {
            "method": "embeddings",
            "results": results,
        }
    else:
        ent_name = text_funcs.get_ent_name(
            entity_id=entity_id, meta_node=meta_node
        )
        # if no such entity
        if ent_name is None:
            results = []
        else:
            indices = get_indices_from_meta_nodes(
                include_meta_nodes, type="text"
            )
            search_res = text_funcs.query_text(
                text=ent_name, client=es_client, indices=indices, limit=limit
            )
            results = text_funcs.format_query_results(search_res)
        res = {
            "method": "text",
            "results": results,
        }
    return res


@router.get("/query/entity/encode", response_model=List[float])
def get_query_ent_encode(
    entity_id: str,
    meta_node: str,
) -> List[float]:
    "Return the text embeddings of the query entity."

    index = meta_node_to_index_name(meta_node)
    embedding_indices = embeddings_funcs.get_embedding_indices(
        client=es_client
    )
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
    embedding_indices = embeddings_funcs.get_embedding_indices(
        client=es_client
    )
    res = [index_name_to_meta_node(_) for _ in embedding_indices]
    return res


def get_indices_from_meta_nodes(meta_nodes: List[str], type: str) -> List[str]:
    if type == "embeddings":
        existing_indices = embeddings_funcs.get_embedding_indices(
            client=es_client
        )
    else:
        existing_indices = text_funcs.get_text_indices(client=es_client)

    if len(meta_nodes) > 0:
        input_indices = [
            meta_node_to_index_name(_, type=type) for _ in meta_nodes
        ]
        indices = list(set(existing_indices).intersection(set(input_indices)))
    else:
        indices = existing_indices
    return indices
