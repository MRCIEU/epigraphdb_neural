# `epigraphdb_neural`

EXPERIMENTAL.

- `main_api`: (docker) main API interfaced by epigraphdb
- `models_api`: (docker) API serving NLP models
  Interfaced by `main_api`.
- `transformers`: (docker) API serving transformer models.
  Interfaced by `models_api`.
- `processing`: (conda) various processing routines

## environment variables

- DOCKER_NEURAL_PORT
- DOCKER_NEURAL_MODELS_PORT
- DOCKER_TRANSFORMERS_PORT
- TRANSFORMERS_URL
- TRANSFORMERS_RAY_NUM_CPUS
