import requests
from fastapi import FastAPI

from epigraphdb_common_utils import neural_env_configs

from .apis import components, nlp, ontology, query

TITLE = "`epigraphdb_neural` API"
app = FastAPI(title=TITLE, docs_url="/")


@app.get("/ping", response_model=bool)
def get_ping(dependencies: bool = True):
    if not dependencies:
        return True
    else:
        models_api_url = neural_env_configs.env_configs["models_api_url"]
        r = requests.get(f"{models_api_url}/ping")
        r.raise_for_status()
        models_api_state = r.json()
        transformers_api_url = neural_env_configs.env_configs[
            "transformers_api_url"
        ]
        r = requests.get(f"{transformers_api_url}/ping")
        r.raise_for_status()
        transformers_api_state = r.json()
        states = [models_api_state, transformers_api_state]
        res = sum(states) == len(states)
        return res


app.include_router(query.router)
app.include_router(nlp.router)
app.include_router(ontology.router)
app.include_router(components.router)
