from fastapi import FastAPI

from .apis import nlp

TITLE = "models API for `epigraphdb_neural` "
app = FastAPI(title=TITLE, docs_url="/")


@app.get("/ping")
def get_ping():
    return True


app.include_router(nlp.router)
