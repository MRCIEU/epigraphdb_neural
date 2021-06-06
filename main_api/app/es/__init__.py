from elasticsearch import Elasticsearch
from loguru import logger

from app.settings import es_host, es_port

es_client = Elasticsearch(
    "http://{host}:{port}".format(host=es_host, port=es_port),
    verify_certs=True,
)


def es_client_connected() -> bool:
    try:
        es_client.ping()
        return True
    except Exception as e:
        logger.error(e)
        return False
