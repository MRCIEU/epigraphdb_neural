from fastapi import FastAPI

from .apis import nlp, query

TITLE = "`epigraphdb_neural` API"
app = FastAPI(title=TITLE, docs_url="/")


@app.get("/ping")
def get_ping():
    return True


app.include_router(query.router)
app.include_router(nlp.router)
