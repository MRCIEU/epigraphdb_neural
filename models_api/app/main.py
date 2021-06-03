from fastapi import FastAPI

from .apis import models

TITLE = "`epigraphdb_neural` API"
app = FastAPI(
    title=TITLE, docs_url="/"
)

@app.get("/ping")
def get_ping():
    return True

app.include_router(models.router)
