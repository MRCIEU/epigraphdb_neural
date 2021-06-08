from epigraphdb_common_utils.neural_env_configs import env_configs

# elasticsearch
es_host = env_configs["es_host"]
es_port = env_configs["es_port"]

# models api
models_api_url = env_configs["models_api_url"]

# params
NUM_ENCODE_LIMIT = 200
MODELS_ENCODE_URL = f"{models_api_url}/nlp/encode"
common_prefix = "embeddings-"
