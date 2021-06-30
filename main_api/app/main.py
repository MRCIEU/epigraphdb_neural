import requests
from fastapi import FastAPI

from epigraphdb_common_utils import neural_env_configs

from .apis import nlp, query

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
        return r.json()


app.include_router(query.router)
app.include_router(nlp.router)
