import json
import sys

import yaml
from transformers import Qwen3VLForConditionalGeneration

from mm_opsd.evaluate import main as evaluate_main
from mm_opsd.io import write_json, write_jsonl
from mm_opsd.modeling import base_model
from mm_opsd.report import main as report_main
from mm_opsd.train import main as train_main


def test_local_pretrained_cli_pipeline(tiny_setup, monkeypatch):
    model, processor, row, config, root = tiny_setup
    base = Qwen3VLForConditionalGeneration(base_model(model).config)
    frozen_state = {name.replace(".base_layer", ""): value
                    for name, value in base_model(model).state_dict().items() if ".lora_" not in name}
    base.load_state_dict(frozen_state)
    pretrained = root / "pretrained"
    base.save_pretrained(pretrained)
    processor.save_pretrained(pretrained)
    config["model"].update(name=str(pretrained), revision=None, min_pixels=4096, max_pixels=4096,
                          lora_rank=2, lora_alpha=4)
    config["train"].update(max_steps=1, validation_samples=1, validation_max_new_tokens=2)
    config["eval"].update(max_new_tokens=2, max_sequence_length=1024)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(root / "manifest.json", {"fixture": "synthetic-image-and-answer"})
    for split in ["train", "validation", "test"]:
        write_jsonl(root / f"geometry3k_{split}.jsonl", [{**row, "id": f"tiny:{split}:0"}])
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--allow-cpu"])
    train_main()
    run = root / "run"
    assert (run / "best_checkpoint.json").exists()
    assert json.loads((run / "validation-step-000001.json").read_text())["count"] == 1
    for name in ["base", "trained"]:
        args = ["evaluate", "--config", str(path), "--benchmark", "geometry3k", "--allow-cpu",
                "--output", str(root / name)]
        if name == "trained":
            args += ["--adapter", str(run)]
        monkeypatch.setattr(sys, "argv", args)
        evaluate_main()
        summary = json.loads((root / name / "summary.json").read_text())
        assert summary["count"] == 1
        assert not summary["official_leaderboard_score"]
    report_path = root / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["report", "--baseline", str(root / "base"),
                                    "--trained", str(root / "trained"), "--output", str(report_path)])
    report_main()
    assert json.loads(report_path.read_text())["count"] == 1
