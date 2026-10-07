import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mm_opsd.config import load_config, validate_config
from mm_opsd.evaluate import evaluate_rows, resolve_adapter, summarize
from mm_opsd.io import dataset_path, file_sha256, read_jsonl, write_json
from mm_opsd.loss import chunked_distillation_loss, chunked_sft_loss
from mm_opsd.modeling import (
    base_model, completion_hidden, correct_answer, encode_prompt, ended_with_eos,
    generation_kwargs, load_model,
)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(model, processor, optimizer, run, state, output, validation=None):
    folder = Path(output) / f"checkpoint-{state['global_step']:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    write_json(folder / "run_config.json", run)
    torch.save({**state, "optimizer": optimizer.state_dict(), "rng": rng_state()},
               folder / "training_state.pt")
    write_json(folder / "state_summary.json", {**state, "validation": validation})
    write_json(Path(output) / "latest_checkpoint.json", {"path": folder.name,
                                                        "step": state["global_step"]})
    best_path = Path(output) / "best_checkpoint.json"
    previous = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
    score = validation["accuracy"] if validation else None
    if previous is None or (score is not None and (previous.get("validation_accuracy") is None
                                                  or score > previous["validation_accuracy"])):
        write_json(best_path, {"path": folder.name, "step": state["global_step"],
                              "validation_accuracy": score})
    return folder


def example_loss(model, processor, row, root, device, config):
    train = config["train"]
    prompt = encode_prompt(processor, row, root, device)
    method = train["method"]
    if method == "sft":
        answer = f"\\boxed{{{correct_answer(row)}}}"
        tokens = processor.tokenizer.encode(answer, add_special_tokens=False)
        eos = base_model(model).generation_config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        if eos is not None:
            tokens.append(eos)
        completion = torch.tensor([tokens], device=device)
        if prompt["input_ids"].shape[1] + len(tokens) > train["max_sequence_length"]:
            raise ValueError(f"{row['id']}: SFT input exceeds max_sequence_length")
        model.train()
        base_model(model).visual.eval()
        hidden = completion_hidden(model, prompt, completion)
        loss = chunked_sft_loss(hidden, completion, base_model(model).lm_head, train["loss_chunk_size"])
        return loss, len(tokens), False

    teacher_prompt = encode_prompt(processor, row, root, device, teacher=method == "opsd")
    budget = min(train["max_completion_length"], train["max_sequence_length"] - max(
        prompt["input_ids"].shape[1], teacher_prompt["input_ids"].shape[1],
    ))
    if budget < 1:
        raise ValueError(f"{row['id']}: image and prompt exceed the training context budget")
    model.eval()
    with torch.no_grad():
        generated = model.generate(**prompt, **generation_kwargs(
            model, budget, sample=True, train_config=train,
            pad_token_id=processor.tokenizer.pad_token_id,
        ))
    completion = generated[:, prompt["input_ids"].shape[1]:]
    if completion.shape[1] == 0:
        raise RuntimeError(f"{row['id']}: model returned an empty rollout")
    truncated = not ended_with_eos(model, completion)
    model.train()
    base_model(model).visual.eval()
    student_hidden = completion_hidden(model, prompt, completion)
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            teacher_hidden = completion_hidden(model, teacher_prompt, completion)
    finally:
        model.train()
        base_model(model).visual.eval()
    loss = chunked_distillation_loss(
        student_hidden, teacher_hidden, base_model(model).lm_head,
        chunk_size=train["loss_chunk_size"], temperature=train["temperature"],
        token_clip=train["token_clip"],
    )
    return loss, completion.shape[1], truncated


def run_training(config, model, processor, rows, validation_rows, device, run, resume=None):
    train = config["train"]
    output = Path(train["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=train["learning_rate"], weight_decay=0.0)
    accumulator = train["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(rows) / accumulator)
    planned_steps = updates_per_epoch * train["epochs"]
    if train["max_steps"] is not None:
        planned_steps = min(planned_steps, train["max_steps"])
    warmup_steps = max(1, math.ceil(run["lr_schedule_steps"] * train["warmup_ratio"]))
    state = {"global_step": 0, "epoch": 0, "next_sample_index": 0, "seen_examples": 0}
    if resume:
        saved = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved.pop("optimizer"))
        restore_rng(saved.pop("rng"))
        state.update(saved)
    write_json(output / "run_config.json", run)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, saved_step = time.monotonic(), state["global_step"] if resume else -1

    def validate_and_save():
        validation = None
        if validation_rows:
            records = evaluate_rows(model, processor, validation_rows, config["data"]["root"], device,
                                    train["validation_max_new_tokens"], config["eval"]["max_sequence_length"])
            validation = summarize(records)
            write_json(output / f"validation-step-{state['global_step']:06d}.json", validation)
        checkpoint = save_checkpoint(model, processor, optimizer, run, state, output, validation)
        print(f"Saved {checkpoint}; validation={validation}", flush=True)

    with (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        if resume:
            log.write(json.dumps({"event": "resume", "from": str(resume), **state}) + "\n")
        progress = tqdm(total=planned_steps, initial=state["global_step"], desc="optimizer updates")
        for epoch in range(state["epoch"], train["epochs"]):
            indices = list(range(len(rows)))
            random.Random(train["seed"] + epoch).shuffle(indices)
            cursor = state["next_sample_index"] if epoch == state["epoch"] else 0
            while cursor < len(indices) and state["global_step"] < planned_steps:
                group = indices[cursor:cursor + accumulator]
                optimizer.zero_grad(set_to_none=True)
                learning_rate = train["learning_rate"] * min(1.0, (state["global_step"] + 1) / warmup_steps)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                group_loss, tokens, truncations = 0.0, 0, 0
                for index in group:
                    loss, count, truncated = example_loss(
                        model, processor, rows[index], config["data"]["root"], device, config,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at {rows[index]['id']}")
                    (loss / len(group)).backward()
                    group_loss += loss.detach().item()
                    tokens += count
                    truncations += truncated
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, train["max_grad_norm"])
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite gradients; inspect data and loss settings")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                cursor += len(group)
                state.update(global_step=state["global_step"] + 1, epoch=epoch,
                             next_sample_index=cursor, seen_examples=state["seen_examples"] + len(group))
                metrics = {
                    **state, "method": train["method"], "loss": group_loss / len(group),
                    "learning_rate": learning_rate, "gradient_norm": float(gradient_norm),
                    "mean_completion_tokens": tokens / len(group),
                    "rollout_truncation_rate": truncations / len(group),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda" else None,
                }
                log.write(json.dumps(metrics) + "\n")
                progress.update()
                progress.set_postfix(loss=f"{metrics['loss']:.5f}", tokens=f"{metrics['mean_completion_tokens']:.1f}")
                if state["global_step"] % train["save_steps"] == 0:
                    validate_and_save()
                    saved_step = state["global_step"]
            if cursor == len(indices):
                state.update(epoch=epoch + 1, next_sample_index=0)
            if state["global_step"] >= planned_steps:
                break
        progress.close()
    if state["global_step"] == 0:
        raise ValueError("Training budget is zero")
    if saved_step != state["global_step"]:
        validate_and_save()
    return state


def main():
    parser = argparse.ArgumentParser(description="Single-GPU multimodal on-policy self-distillation")
    parser.add_argument("--config", default="configs/l20.yaml")
    parser.add_argument("--method", choices=["opsd", "sft", "no_answer"])
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("Use python/mmopsd-train directly; this implementation targets one GPU.")
    config = load_config(args.config)
    for key in ["method", "output_dir", "max_steps", "max_train_samples", "epochs"]:
        if getattr(args, key) is not None:
            config["train"][key] = getattr(args, key)
    validate_config(config)
    output = Path(config["train"]["output_dir"])
    if not args.resume_from and output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty. Resume explicitly or choose a new --output-dir.")
    manifest = Path(config["data"]["root"]) / "manifest.json"
    if not manifest.exists():
        raise SystemExit("Prepared data is missing. Run mmopsd-prepare first.")
    source = dataset_path(config["data"]["root"], "geometry3k", "train")
    rows = read_jsonl(source)
    if config["train"]["max_train_samples"]:
        rows = rows[:config["train"]["max_train_samples"]]
    if not rows:
        raise ValueError("Training dataset is empty")
    validation_rows = read_jsonl(dataset_path(config["data"]["root"], "geometry3k", "validation"))
    validation_rows = validation_rows[:config["train"]["validation_samples"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("CUDA is required; --allow-cpu is only for debugging small models.")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("The planned BF16 recipe requires a BF16-capable GPU.")
    random.seed(config["train"]["seed"])
    torch.manual_seed(config["train"]["seed"])
    resume, previous = None, None
    if args.resume_from:
        resume = resolve_adapter(args.resume_from, "latest")
        previous = json.loads((resume / "run_config.json").read_text(encoding="utf-8"))
        old_config = previous["config"]
        for section in ["model", "data"]:
            if old_config[section] != config[section]:
                raise ValueError(f"Resume configuration changed: {section}")
        mutable = {"epochs", "max_steps", "output_dir"}
        for key, value in old_config["train"].items():
            if key not in mutable and value != config["train"][key]:
                raise ValueError(f"Resume training configuration changed: {key}")
        if previous["training_data_sha256"] != file_sha256(source) or previous["manifest_sha256"] != file_sha256(manifest):
            raise ValueError("Prepared data changed since the checkpoint")
    model, processor, revision = load_model(
        config, device, adapter=resume, trainable=True,
        revision=previous["model_revision"] if previous else None,
    )
    steps = math.ceil(len(rows) / config["train"]["gradient_accumulation_steps"]) * config["train"]["epochs"]
    steps = min(steps, config["train"]["max_steps"]) if config["train"]["max_steps"] else steps
    run = {
        "config": config, "model_revision": revision, "training_data_sha256": file_sha256(source),
        "manifest_sha256": file_sha256(manifest), "training_examples": len(rows),
        "validation_examples": len(validation_rows), "lr_schedule_steps": previous["lr_schedule_steps"] if previous else steps,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
    }
    state = run_training(config, model, processor, rows, validation_rows, device, run, resume)
    print(json.dumps({"finished": state, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
