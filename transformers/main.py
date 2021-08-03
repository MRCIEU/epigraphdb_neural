from pathlib import Path
from typing import List

import environs
import ray
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class InferenceRequest(BaseModel):
    text_1: List[str]
    text_2: List[str]


MAX_NUM_TEXT_PAIRS = 20
BASE_MODEL_NAME = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
MAX_LENGTH = 32
MODEL_PATH = Path("/models") / "bluebert-efo" / "pytorch_model.bin"
CONFIG_PATH = MODEL_PATH.parent / "config.json"
assert MODEL_PATH.exists()
assert CONFIG_PATH.exists()

env = environs.Env()
env.read_env()
NUM_CPUS = env.int("TRANSFORMERS_RAY_NUM_CPUS", 4)
print(f"NUM_CPUS: {NUM_CPUS}")

app = FastAPI(docs_url="/")
serve_handle = None


@app.get("/ping", response_model=bool)
async def get_ping():
    """
    Returns True when the service is online.
    """
    return True


class BlueBertInference:
    def __init__(self):
        self.config = AutoConfig.from_pretrained(
            str(CONFIG_PATH), num_labels=1
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH), config=self.config
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print("Model init complete.")

    def inference(self, text_1: List[str], text_2: List[str]) -> List[float]:
        encodings = self.tokenizer(
            text_1,
            text_2,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.model(**encodings)["logits"].reshape(-1).tolist()
        return output

    async def __call__(self, request: serve.utils.ServeRequest) -> List[float]:
        request_data: InferenceRequest = await request.body()
        text_1 = request_data.text_1
        text_2 = request_data.text_2
        # inference
        inference_output: List[float] = self.inference(text_1, text_2)
        return inference_output


@app.on_event("startup")
async def startup_event():
    ray.init(num_cpus=NUM_CPUS)
    client = serve.start(http_host=None)

    # Set up a Ray Serve backend with the desired number of replicas.
    backend_config = serve.BackendConfig(num_replicas=1)
    ray_actor_options = {"num_cpus": 1, "num_gpus": 0}
    client.create_backend(
        "bluebert",
        BlueBertInference,
        config=backend_config,
        ray_actor_options=ray_actor_options,
    )
    client.create_endpoint("inference", backend="bluebert")

    # Get a handle to our Ray Serve endpoint so we can query it in Python.
    global serve_handle
    serve_handle = client.get_handle("inference")


@app.post("/inference", response_model=List[float])
async def inference(request_data: InferenceRequest):
    return await serve_handle.remote(request_data)  # type: ignore
