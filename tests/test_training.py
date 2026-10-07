import random

import pytest
import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

from mm_opsd.modeling import base_model
from mm_opsd.train import example_loss, restore_rng, rng_state, run_training


@pytest.mark.parametrize("method", ["opsd", "sft", "no_answer"])
def test_training_methods_have_finite_loss(tiny_setup, method):
    model, processor, row, config, root = tiny_setup
    config["train"]["method"] = method
    loss, tokens, truncated = example_loss(model, processor, row, root, torch.device("cpu"), config)
    assert torch.isfinite(loss)
    assert tokens > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_training_checkpoint_and_optimizer_resume(tiny_setup):
    model, processor, row, config, root = tiny_setup
    config["train"]["method"] = "sft"
    config["train"]["max_steps"] = 1
    run = {"config": config, "model_revision": None, "lr_schedule_steps": 2}
    state = run_training(config, model, processor, [row, row], [], torch.device("cpu"), run)
    checkpoint = root / "run" / "checkpoint-000001"
    assert state["global_step"] == 1
    assert (checkpoint / "training_state.pt").exists()
    saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    assert saved["next_sample_index"] == 1
    assert saved["optimizer"]["state"]
    trained_adapters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                        if parameter.requires_grad}
    base = Qwen3VLForConditionalGeneration(base_model(model).config)
    frozen_state = {name.replace(".base_layer", ""): value for name, value in base_model(model).state_dict().items()
                    if ".lora_" not in name}
    base.load_state_dict(frozen_state)
    restored = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    for name, parameter in restored.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, trained_adapters[name])
    config["train"]["max_steps"] = 2
    state = run_training(config, restored, processor, [row, row], [], torch.device("cpu"), run, checkpoint)
    assert state["global_step"] == 2
    assert state["seen_examples"] == 2
    assert (root / "run" / "checkpoint-000002" / "adapter_model.safetensors").exists()


def test_rng_restore():
    random.seed(9)
    torch.manual_seed(9)
    state = rng_state()
    expected = (random.random(), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1])
