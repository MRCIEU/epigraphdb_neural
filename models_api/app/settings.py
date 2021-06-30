import environs

env = environs.Env()
env.read_env()

# default to same docker-compose
TRANSFORMERS_URL = env("TRANSFORMERS_URL", "http://transformers:8000")
