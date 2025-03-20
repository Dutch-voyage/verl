import os
import sys
import torch
import pytest
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray import (
    RayWorkerGroup,
    RayResourcePool,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.ray_trainer import StatefulDataLoader
from verl.utils.dataset.rl_dataset import RLHFDataset
from torch.utils.data import DataLoader
from verl.trainer.ppo.ray_trainer import collate_fn
import time
from verl.workers.reward_manager import NaiveRewardManager
from torch.utils.data import SequentialSampler
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.cce_actor import CCE_DP_PPOActor
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo import core_algos


def train_cce(config, model, optimizer, batch):
    actor_cce = CCE_DP_PPOActor(config.actor_rollout_ref.actor, model, optimizer)
    batch = batch.to(torch.cuda.current_device())
    old_log_probs = actor_cce.compute_log_prob(batch)
    olp_batch = DataProto.from_dict(
        tensors={"old_log_probs": old_log_probs},
        meta_info={"temperature": config.actor_rollout_ref.rollout.temperature},
    )
    batch = batch.union(olp_batch)
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
    batch = batch.select(batch_keys=select_keys).batch
    batch = batch.split(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    batch = batch[0]
    batch = batch.to(torch.cuda.current_device())
    
    entropy, log_prob = actor_cce._forward_micro_batch(
        micro_batch=batch, temperature=config.actor_rollout_ref.rollout.temperature
    )

    old_log_prob = batch["old_log_probs"]
    advantages = batch["advantages"]
    attention_mask = batch['attention_mask']
    response_length = batch['responses'].size(1)
    response_mask = attention_mask[:, -response_length:]
    clip_ratio = config.actor_rollout_ref.actor.clip_ratio
    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=response_mask,
        cliprange=clip_ratio,
    )
    pg_loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    # optimizer.step()
    optimizer.zero_grad()
    return pg_loss, grads


def train_dp(config, model, optimizer, batch):
    actor_dp = DataParallelPPOActor(config.actor_rollout_ref.actor, model, optimizer)
    batch = batch.to(torch.cuda.current_device())
    old_log_probs = actor_dp.compute_log_prob(batch)
    olp_batch = DataProto.from_dict(
        tensors={"old_log_probs": old_log_probs},
        meta_info={"temperature": config.actor_rollout_ref.rollout.temperature},
    )
    batch = batch.union(olp_batch)
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
    batch = batch.select(batch_keys=select_keys).batch
    batch = batch.split(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    batch = batch[0]
    
    old_log_prob = batch["old_log_probs"]
    advantages = batch["advantages"]
    attention_mask = batch['attention_mask']
    response_length = batch['responses'].size(1)
    response_mask = attention_mask[:, -response_length:]
    
    entropy, log_prob = actor_dp._forward_micro_batch(
        micro_batch=batch, temperature=config.actor_rollout_ref.rollout.temperature
    )

    clip_ratio = config.actor_rollout_ref.actor.clip_ratio
    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=response_mask,
        cliprange=clip_ratio,
    )
    pg_loss.backward() 
    grads = {name: p.grad for name, p in model.named_parameters()}
    optimizer.zero_grad()
    return pg_loss, grads

def get_batch(config):
    batch = DataProto.load_from_disk("../../my-Logic/test_e2e_data.pkl")
    batch = batch.chunk(4)[0]
    batch = compute_advantage(
        batch, adv_estimator="reinforce_plus_plus", gamma=1.0, lam=1.0, num_repeat=16
    )
    batch.meta_info["micro_batch_size"] = (
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
    )
    batch.meta_info["max_token_len"] = (
        config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
    )
    batch.meta_info["use_dynamic_bsz"] = False
    batch.meta_info["temperature"] = config.actor_rollout_ref.rollout.temperature
    return batch

def test_bwd():
    model_path = "/home/yyx/models/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.actor.use_remove_padding = True
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 160
    config.actor_rollout_ref.model.use_remove_padding = True
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    
    batch_cce = get_batch(config)
    batch_dp = get_batch(config)
    
    loss_cce, grad_cce = train_cce(config, model, optimizer, batch_cce)
    loss_dp, grad_dp = train_dp(config, model, optimizer, batch_dp)
    
    print(loss_cce, loss_dp)
    print(grad_dp['model.embed_tokens.weight'])
    print(grad_cce['model.embed_tokens.weight'])
    difference = torch.sum(torch.abs(grad_dp['model.embed_tokens.weight'] - grad_cce['model.embed_tokens.weight']))
    print(difference)

if __name__ == "__main__":
    test_bwd()
