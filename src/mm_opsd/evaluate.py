import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from mm_opsd.config import load_config
from mm_opsd.io import dataset_path, file_sha256, read_jsonl, write_json, write_jsonl
from mm_opsd.modeling import encode_prompt, ended_with_eos, generation_kwargs, load_model
from mm_opsd.scoring import score_answer


def resolve_adapter(path, checkpoint="best"):
    path = Path(path)
    pointer = path / f"{checkpoint}_checkpoint.json"
    if pointer.exists():
        info = json.loads(pointer.read_text(encoding="utf-8"))
        return path / info["path"]
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter or {checkpoint} checkpoint pointer at {path}")
    return path


def evaluate_rows(model, processor, rows, root, device, max_new_tokens, max_sequence_length):
    was_training = model.training
    model.eval()
    records = []
    try:
        with torch.inference_mode():
            for row in tqdm(rows, desc="evaluate", leave=False):
                prompt = encode_prompt(processor, row, root, device)
                prompt_length = prompt["input_ids"].shape[1]
                budget = min(max_new_tokens, max_sequence_length - prompt_length)
                if budget < 1:
                    raise ValueError(f"{row['id']}: image and question exceed evaluation context budget")
                generated = model.generate(**prompt, **generation_kwargs(
                    model, budget, pad_token_id=processor.tokenizer.pad_token_id,
                ))
                completion = generated[:, prompt_length:]
                text = processor.tokenizer.decode(completion[0], skip_special_tokens=True)
                score = score_answer(text, row)
                records.append({
                    "id": row["id"], "response": text, "answer": row["answer"], **score,
                    "generated_tokens": completion.shape[1],
                    "truncated": not ended_with_eos(model, completion),
                    "generation_budget": budget,
                })
    finally:
        model.train(was_training)
        if was_training:
            from mm_opsd.modeling import base_model
            base_model(model).visual.eval()
    return records


def summarize(records):
    count = len(records)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {
        "count": count, "correct": sum(item["correct"] for item in records),
        "accuracy": sum(item["correct"] for item in records) / count,
        "parse_failure_rate": sum(item["parse_failed"] for item in records) / count,
        "truncation_rate": sum(item["truncated"] for item in records) / count,
        "mean_generated_tokens": sum(item["generated_tokens"] for item in records) / count,
    }


def main():
    parser = argparse.ArgumentParser(description="Greedy evaluation with local, deterministic answer scoring")
    parser.add_argument("--config", default="configs/l20.yaml")
    parser.add_argument("--benchmark", choices=["geometry3k", "mathvista", "mathvision"], required=True)
    parser.add_argument("--split")
    parser.add_argument("--adapter")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    for key in ["max_samples", "max_new_tokens"]:
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Evaluation output is not empty; choose a new --output to preserve prior results.")
    config = load_config(args.config)
    split = args.split or ("test" if args.benchmark == "geometry3k" else "testmini")
    source = dataset_path(config["data"]["root"], args.benchmark, split)
    rows = read_jsonl(source)
    if args.max_samples:
        rows = rows[:args.max_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("CUDA is required; --allow-cpu is for debugging only.")
    adapter = resolve_adapter(args.adapter, args.checkpoint) if args.adapter else None
    revision = None
    if adapter:
        training = json.loads((adapter / "run_config.json").read_text(encoding="utf-8"))
        if training["config"]["model"]["name"] != config["model"]["name"]:
            raise ValueError("Adapter and evaluation base model do not match")
        revision = training["model_revision"]
        for key in ["max_pixels", "min_pixels"]:
            if training["config"]["model"][key] != config["model"][key]:
                raise ValueError(f"Evaluation {key} differs from the training configuration")
    model, processor, revision = load_model(config, device, adapter=adapter, revision=revision)
    max_new_tokens = args.max_new_tokens or config["eval"]["max_new_tokens"]
    records = evaluate_rows(model, processor, rows, config["data"]["root"], device,
                            max_new_tokens, config["eval"]["max_sequence_length"])
    metadata = {
        "benchmark": args.benchmark, "split": split, "dataset_sha256": file_sha256(source),
        "model": config["model"]["name"], "model_revision": revision,
        "adapter": str(adapter) if adapter else None, "max_pixels": config["model"]["max_pixels"],
        "min_pixels": config["model"]["min_pixels"], "max_new_tokens": max_new_tokens,
        "max_sequence_length": config["eval"]["max_sequence_length"],
        "decoding": "greedy", "scoring": "mm-opsd-local-v1",
        "official_leaderboard_score": False,
    }
    write_jsonl(output / "predictions.jsonl", records)
    summary = {**metadata, **summarize(records)}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
