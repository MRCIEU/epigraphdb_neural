import pytest

from app.apis import query

text_queries = [
    ("Body mass index"),
    ("Coronary heart disease"),
    ("Obesity"),
    ("Asthma"),
]

entity_queries = [
    ("Gwas", "ieu-a-2"),
    ("Gwas", "ieu-a-90"),
    ("Efo", "http://www.ebi.ac.uk/efo/EFO_0001073"),
    ("Disease", "http://purl.obolibrary.org/obo/HP_0001513"),
]


@pytest.mark.parametrize("text", text_queries)
def test_get_query_text(text):
    for method in query.models.QueryMethodOptions:
        res = query.get_query_text(
            text=text,
            asis=True,
            limit=5,
            include_meta_nodes=[],
            method=method,
        )
        assert res is not None
        assert len(res["results"]) > 0


@pytest.mark.parametrize("meta_node, entity_id", entity_queries)
def test_get_query_ent(meta_node, entity_id):
    for method in query.models.QueryMethodOptions:
        res = query.get_query_ent(
            entity_id=entity_id,
            meta_node=meta_node,
            limit=5,
            include_meta_nodes=[],
            method=method,
        )
        assert res is not None
        assert len(res["results"]) > 0
