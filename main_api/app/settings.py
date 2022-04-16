from epigraphdb_common_utils.neural_env_configs import env_configs

# elasticsearch
es_host = env_configs["es_host"]
es_port = env_configs["es_port"]

# models api
models_api_url = env_configs["models_api_url"]
transformers_api_url = env_configs["transformers_api_url"]

# params
NUM_ENCODE_LIMIT = 200
NUM_SIM_LIMIT = 9
NUM_ONTOLOGY_DISTANCE_LIMIT = 10
MODELS_ENCODE_URL = f"{models_api_url}/nlp/encode"
TRANSFORMERS_INFERENCE_URL = f"{transformers_api_url}/inference"
embeddings_common_prefix = "embeddings-"
# NOTE: this logic currently sits in epigraphdb backend
# TODO: implement this logic at neural instead
text_common_prefix = "search-global-"
