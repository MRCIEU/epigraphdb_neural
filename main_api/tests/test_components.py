import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

params = [
    # status
    ("models_api", "/ping", "GET", None),
    ("transformers_api", "/ping", "GET", None),
    ("models_api", "/nlp/model-info", "GET", None),
    # transformers
    (
        "transformers_api",
        "/inference",
        "POST",
        {
            "text_1": ["body mass index", "obesity"],
            "text_2": ["obesity", "body mass index"],
        },
    ),
    # models api
    (
        "models_api",
        "/nlp/encode",
        "GET",
        {"text": "body mass index", "asis": False, "nlp_model": "scispacy_lg"},
    ),
    (
        "models_api",
        "/nlp/encode",
        "GET",
        {"text": "body mass index", "asis": True, "nlp_model": "default"},
    ),
    (
        "models_api",
        "/nlp/encode",
        "POST",
        {
            "text_list": ["body mass index", "obesity"],
            "asis": False,
            "nlp_model": "scispacy_lg",
        },
    ),
    (
        "models_api",
        "/nlp/encode",
        "POST",
        {
            "text_list": ["body mass index", "obesity"],
            "asis": True,
            "nlp_model": "default",
        },
    ),
    (
        "models_api",
        "/nlp/similarity",
        "GET",
        {"text1": "body mass index", "text2": "body", "nlp_model": "default"},
    ),
    (
        "models_api",
        "/nlp/similarity",
        "GET",
        {"text1": "obesity", "text2": "body", "nlp_model": "scispacy_lg"},
    ),
]


@pytest.mark.parametrize("component, route, method, request_payload", params)
def test_responses(component, route, method, request_payload):
    print(locals())
    url = "/components"
    payload = {
        "component": component,
        "route": route,
        "method": method,
        "payload": request_payload,
    }
    r = client.post(url, json=payload)
    assert r.ok
