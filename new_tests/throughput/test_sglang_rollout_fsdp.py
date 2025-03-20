# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import sys
import torch
import pytest
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

# Skip test if not enough GPUs
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs")
def test_sglang_rollout_fsdp():
    # os.environ['VLLM_USE_V1'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = "0,1"

    # Initialize ray
    if not ray.is_initialized():
        ray.init()
    
    # Set up model path - replace with appropriate test model
    model_path = os.environ.get("TEST_MODEL_PATH", "/home/yyx/models/Qwen2.5-0.5B")
    
    # Create config
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config = config.actor_rollout_ref
    config.model.path = model_path
    config.rollout.name = "sglang"
    config.rollout.tensor_model_parallel_size = 1
    config.rollout.response_length = 256
    # Create a resource pool directly
    resource_pool = RayResourcePool(
        process_on_nodes=[2],  # 2 GPUs on 1 node
        use_gpu=True,
        max_colocate_count=1,
        name_prefix="test_pool"
    )
    
    actor_rollout_cls = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker),
                                             config=config,
                                             role='actor_rollout')
    all_wg = {}
    class_dict = {"actor_rollout": actor_rollout_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
    worker = all_wg['actor_rollout']
    worker.init_model()
    # Create test prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_prompts = [
        "Write a short poem about AI.",
        "Explain quantum computing in simple terms.",
        "What is the capital of France?",
        "Why is the sky blue?",
    ]
    
    encoded_prompts = tokenizer(test_prompts, return_tensors="pt", padding=True)
    input_ids = encoded_prompts["input_ids"]
    attention_mask = encoded_prompts["attention_mask"]
    
    # Pad to max length
    # max_prompt_length = config.rollout.prompt_length
    max_prompt_length = 256
    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)
    
    # Create position ids
    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    
    # Create DataProto
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    
    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    prompts = DataProto.from_dict(batch, meta_info=meta_info)
    
    # Generate sequences
    print("Generating sequences...")
    outputs = worker.generate_sequences(prompts)
    
    # Verify outputs
    assert outputs is not None
    assert "responses" in outputs.batch
    
    # Decode responses
    responses = outputs.batch["responses"]
    decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
    
    # Print responses for inspection
    print("Generated responses:")
    for i, resp in enumerate(decoded_responses):
        print(f"Prompt {i+1}: {test_prompts[i]}")
        print(f"Response: {resp}")
        print("-" * 50)
    
    # Verify responses are not empty
    for resp in decoded_responses:
        assert len(resp.strip()) > 0
    
    # Clean up
    # ray.kill(worker)
    ray.shutdown()

if __name__ == "__main__":
    test_sglang_rollout_fsdp()