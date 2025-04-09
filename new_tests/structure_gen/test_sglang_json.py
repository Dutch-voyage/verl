import os
import sys
import torch
import pytest
import ray
from omegaconf import OmegaConf
from sglang.srt.entrypoints import engine
from pydantic import BaseModel, Field
import json

class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")

def test_sglang_json():
    model_path = os.environ.get("TEST_MODEL_PATH", "/home/yyx/models/Qwen2.5-0.5B")
    sampling_params = {
        "temperature": 0.1,
        "top_p": 0.95,
        "json_schema": json.dumps(CapitalInfo.model_json_schema()),
    }
    llm = engine.Engine(model_path=model_path)
    prompts = [
        "Give me the information of the capital of China in the JSON format.",
    ]
    outputs = llm.generate(
        prompt=prompts,
        sampling_params=sampling_params,
    )
    print(outputs)

if __name__ == "__main__":
    test_sglang_json()