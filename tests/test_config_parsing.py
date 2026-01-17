#!/usr/bin/env python3
"""
Tests for config parsing and validation.
"""

from argparse import Namespace

import pytest

from halo_forge.cli import cmd_config_validate, _load_prompts_jsonl
from halo_forge.sft.trainer import SFTConfig


def _write_yaml(path, content: str) -> str:
    path.write_text(content)
    return str(path)


def test_sft_from_yaml_reads_lora(tmp_path):
    cfg_path = tmp_path / "sft.yaml"
    _write_yaml(
        cfg_path,
        """
model:
  name: Qwen/Qwen2.5-Coder-7B
data:
  train_file: data/train.jsonl
lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: [q_proj, k_proj]
training:
  output_dir: models/sft
  per_device_train_batch_size: 4
  num_train_epochs: 2
""",
    )

    cfg = SFTConfig.from_yaml(str(cfg_path))
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
    assert cfg.target_modules == ["q_proj", "k_proj"]
    assert cfg.batch_size == 4
    assert cfg.num_epochs == 2


def test_cmd_config_validate_sft_nested(tmp_path):
    cfg_path = tmp_path / "sft.yaml"
    _write_yaml(
        cfg_path,
        """
model:
  name: Qwen/Qwen2.5-Coder-7B
data:
  train_file: data/train.jsonl
training:
  output_dir: models/sft
  learning_rate: 2e-4
""",
    )

    args = Namespace(config=str(cfg_path), type="sft", verbose=False)
    cmd_config_validate(args)


def test_cmd_config_validate_raft_nested(tmp_path):
    cfg_path = tmp_path / "raft.yaml"
    _write_yaml(
        cfg_path,
        """
output_dir: models/raft
prompts: data/prompts.jsonl
raft:
  num_cycles: 3
  reward_threshold: 0.5
generation:
  temperature: 0.7
""",
    )

    args = Namespace(config=str(cfg_path), type="raft", verbose=False)
    cmd_config_validate(args)


def test_load_prompts_jsonl_invalid(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt": "ok"}\n'
        '{"prompt": "still ok"}\n'
        '{invalid json}\n'
    )

    prompts, invalid = _load_prompts_jsonl(str(prompts_path))
    assert prompts == ["ok", "still ok"]
    assert len(invalid) == 1

