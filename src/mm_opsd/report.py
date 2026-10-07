import argparse
import json
import math
import random
from pathlib import Path

from mm_opsd.io import read_jsonl, write_json


def paired_statistics(baseline, trained, repetitions=1000, seed=42):
    if [row["id"] for row in baseline] != [row["id"] for row in trained]:
        raise ValueError("Comparisons require exactly the same ordered question IDs.")
    changes = [int(right["correct"]) - int(left["correct"])
               for left, right in zip(baseline, trained)]
    if not changes:
        raise ValueError("Cannot compare empty evaluations")
    improved, regressed = changes.count(1), changes.count(-1)
    disagreements = improved + regressed
    p_value = min(1.0, 2 * sum(math.comb(disagreements, i)
                              for i in range(min(improved, regressed) + 1)) / (1 << disagreements))
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choices(changes, k=len(changes))) / len(changes)
                       for _ in range(repetitions))
    return {
        "count": len(changes), "accuracy_delta": sum(changes) / len(changes),
        "improved_questions": improved, "regressed_questions": regressed,
        "paired_bootstrap_95_ci": [bootstrap[int(0.025 * repetitions)],
                                    bootstrap[min(repetitions - 1, int(0.975 * repetitions))]],
        "mcnemar_exact_p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare paired evaluation outputs")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    left_path, right_path = Path(args.baseline), Path(args.trained)
    left = json.loads((left_path / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((right_path / "summary.json").read_text(encoding="utf-8"))
    for key in ["benchmark", "split", "dataset_sha256", "model", "model_revision",
                "max_pixels", "min_pixels", "max_new_tokens", "max_sequence_length", "decoding", "scoring"]:
        if left[key] != right[key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    result = {
        "benchmark": left["benchmark"], "baseline_accuracy": left["accuracy"],
        "trained_accuracy": right["accuracy"],
        **paired_statistics(read_jsonl(left_path / "predictions.jsonl"),
                            read_jsonl(right_path / "predictions.jsonl"), args.bootstrap_samples),
        "baseline_parse_failure_rate": left["parse_failure_rate"],
        "trained_parse_failure_rate": right["parse_failure_rate"],
        "baseline_truncation_rate": left["truncation_rate"],
        "trained_truncation_rate": right["truncation_rate"],
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
