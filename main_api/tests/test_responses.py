import pytest
from icecream import ic
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

params = [
    ("/ping", "GET", None),
    ("/nlp/encode/text", "GET", {"text": "Body mass index"}),
    (
        "/nlp/encode/text",
        "POST",
        {
            "text_list": [
                "Body mass index",
                "Coronary heart disease",
                "years of schooling",
            ]
        },
    ),
    (
        "/nlp/similarity/text",
        "POST",
        {"text_list": ["Body mass index", "Body weight", "Obesity"]},
    ),
    ("/query/text", "GET", {"text": "Body mass index"}),
    ("/query/entity", "GET", {"entity_id": "ieu-a-2", "meta_node": "Gwas"}),
    ("/query/meta-entity-list", "GET", None),
    (
        "/ontology/distance",
        "POST",
        {
            "text_1": ["body mass index", "obesity"],
            "text_2": ["obesity", "body mass index"],
        },
    ),
]


@pytest.mark.parametrize("url, method, params", params)
def test_responses(url, method, params):
    ic(url, method, params)
    if method == "GET":
        r = client.get(url, params=params)
    elif method == "POST":
        r = client.post(url, json=params)
    assert r.ok
