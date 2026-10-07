from pathlib import Path

import torch
from PIL import Image


LANGUAGE_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def load_model(config, device, adapter=None, trainable=False, revision=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    spec = config["model"]
    revision = revision or spec.get("revision")
    if not Path(spec["name"]).is_dir() and revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(spec["name"]).sha
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        spec["name"], revision=revision, dtype=dtype, attn_implementation=spec["attention"],
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        spec["name"], revision=revision, min_pixels=spec["min_pixels"],
        max_pixels=spec["max_pixels"], use_fast=False,
    )
    base.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(base, LoraConfig(
            r=spec["lora_rank"], lora_alpha=spec["lora_alpha"], lora_dropout=0.0,
            target_modules=LANGUAGE_LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        ))
    else:
        model = base
    if trainable:
        if any(parameter.requires_grad for parameter in base.visual.parameters()):
            raise RuntimeError("Visual weights must remain frozen for the shared fixed teacher.")
        if config["train"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    return model, processor, revision


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def correct_answer(row):
    if row["choices"]:
        answer = row["answer"].strip()
        if answer in row["choices"]:
            return chr(65 + row["choices"].index(answer))
        if len(answer) == 1 and "A" <= answer.upper() < chr(65 + len(row["choices"])):
            return answer.upper()
    return row["answer"]


def prompt_text(row, teacher=False):
    text = f"Problem: {row['question']}"
    if row["choices"]:
        text += "\nChoices:\n" + "\n".join(
            f"({chr(65 + index)}) {choice}" for index, choice in enumerate(row["choices"])
        )
    if teacher:
        text += (
            f"\n\nThe verified final answer is: {correct_answer(row)}. "
            "No reference reasoning is provided. Use the image and the problem to derive "
            "this result through your own reasoning."
        )
    instruction = "Give concise step-by-step reasoning and put only your final answer in \\boxed{} ."
    if row["choices"]:
        instruction += " The final answer must be the option letter."
    elif row.get("answer_type") == "float" and row.get("precision") is not None:
        instruction += f" Round the final number to {row['precision']} decimal places."
    return text + "\n\n" + instruction


def encode_prompt(processor, row, root, device, teacher=False):
    with Image.open(Path(root) / row["image"]) as handle:
        image = handle.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text(row, teacher)},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors="pt", padding=False)
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def attach_completion(prompt, completion):
    result = dict(prompt)
    result["input_ids"] = torch.cat([prompt["input_ids"], completion], dim=1)
    result["attention_mask"] = torch.cat([
        prompt["attention_mask"], torch.ones_like(completion),
    ], dim=1)
    return result


def completion_hidden(model, prompt, completion):
    base = base_model(model)
    full = attach_completion(prompt, completion)
    # Teacher and student have different text offsets; never reuse cached multimodal positions.
    positions, _ = base.model.get_rope_index(
        full["input_ids"], image_grid_thw=full.get("image_grid_thw"),
        attention_mask=full["attention_mask"],
    )
    output = base.model(**full, position_ids=positions, use_cache=False, return_dict=True)
    start = prompt["input_ids"].shape[1] - 1
    return output.last_hidden_state[:, start:start + completion.shape[1], :]


def generation_kwargs(model, max_new_tokens, sample=False, train_config=None, pad_token_id=None):
    base = base_model(model)
    forbidden = [getattr(base.config, key, None) for key in (
        "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
    )]
    kwargs = {
        "max_new_tokens": max_new_tokens, "do_sample": sample, "use_cache": True,
        "suppress_tokens": sorted({token for token in forbidden if token is not None}),
        "pad_token_id": pad_token_id, "eos_token_id": base.generation_config.eos_token_id,
    }
    if sample:
        kwargs.update(temperature=train_config["temperature"], top_p=train_config["top_p"],
                      top_k=train_config["top_k"])
    else:
        kwargs.update(temperature=None, top_p=None, top_k=None)
    return kwargs


def ended_with_eos(model, completion):
    eos = base_model(model).generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    return completion.shape[1] > 0 and completion[0, -1].item() in eos
