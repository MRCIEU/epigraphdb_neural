import requests
from fastapi import FastAPI

from .apis import nlp
from .settings import TRANSFORMERS_URL

TITLE = "models API for `epigraphdb_neural` "
app = FastAPI(title=TITLE, docs_url="/")


@app.get("/ping", response_model=bool)
def get_ping(dependencies: bool = True):
    if not dependencies:
        return True
    else:
        r = requests.get(f"{TRANSFORMERS_URL}/ping")
        r.raise_for_status()
        return r.json()


app.include_router(nlp.router)
