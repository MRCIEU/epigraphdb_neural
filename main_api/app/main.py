from fastapi import FastAPI

TITLE = "`epigraphdb_neural` API"
app = FastAPI(title=TITLE, docs_url="/")


@app.get("/ping")
def get_ping():
    return True
