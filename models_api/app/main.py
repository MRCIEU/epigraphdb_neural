from fastapi import FastAPI

TITLE = "Models API for `epigraphdb_neural`"
app = FastAPI(
    title=TITLE, docs_url="/"
)

@app.get("/ping")
def get_ping():
    return True
