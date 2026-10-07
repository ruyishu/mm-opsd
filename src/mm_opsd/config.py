from copy import deepcopy
from pathlib import Path

import yaml


def merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent), config)
    validate_config(config)
    return config


def validate_config(config):
    if config["train"]["batch_size"] != 1:
        raise ValueError("The single-GPU trainer supports batch_size=1; use gradient accumulation.")
    if config["train"]["method"] not in {"opsd", "sft", "no_answer"}:
        raise ValueError("method must be opsd, sft, or no_answer")
    if config["train"]["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config["train"]["loss_chunk_size"] < 1:
        raise ValueError("loss_chunk_size must be positive")
    if config["model"]["min_pixels"] > config["model"]["max_pixels"]:
        raise ValueError("min_pixels cannot exceed max_pixels")
    for key in ["epochs", "save_steps", "max_sequence_length", "max_completion_length",
                "validation_max_new_tokens", "temperature", "learning_rate", "max_grad_norm"]:
        if config["train"][key] <= 0:
            raise ValueError(f"train.{key} must be positive")
    for key in ["max_steps", "max_train_samples"]:
        if config["train"][key] is not None and config["train"][key] <= 0:
            raise ValueError(f"train.{key} must be positive or null")
    if config["train"]["validation_samples"] < 0:
        raise ValueError("validation_samples cannot be negative")
    for key in ["max_sequence_length", "max_new_tokens"]:
        if config["eval"][key] <= 0:
            raise ValueError(f"eval.{key} must be positive")
