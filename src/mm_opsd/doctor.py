import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys

import torch


EXPECTED = {
    "torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.1",
    "trl": "0.26.0", "accelerate": "1.11.0", "peft": "0.17.1", "datasets": "3.6.0",
    "qwen-vl-utils": "0.0.14", "Pillow": "11.3.0", "PyYAML": "6.0.2",
    "math-verify": "0.8.0", "safetensors": "0.5.3", "numpy": "1.26.4",
    "huggingface-hub": "0.36.0", "wandb": "0.22.3",
}


def main():
    parser = argparse.ArgumentParser(description="Check the pinned environment and BF16 GPU execution")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    versions, errors = {}, []
    for package, expected in EXPECTED.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        versions[package] = version
        if version is None or version.split("+")[0] != expected:
            errors.append(f"{package}: expected {expected}, found {version}")
    result = {"python": sys.version.split()[0], "platform": platform.platform(),
              "versions": versions, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if not args.allow_cpu and sys.version_info[:2] != (3, 10):
        errors.append("Use the planned Python 3.10 environment (environment.yml pins 3.10.16).")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": properties.name, "total_gib": properties.total_memory / 2**30,
                         "compute_capability": [properties.major, properties.minor],
                         "bf16_supported": torch.cuda.is_bf16_supported()}
        if not result["gpu"]["bf16_supported"]:
            errors.append("Native BF16 is required for this recipe")
        if torch.version.cuda != "12.8":
            errors.append("Expected the PyTorch cu128 build")
        try:
            tensor = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            (tensor @ tensor.T).float().square().mean().backward()
            torch.cuda.synchronize()
            result["bf16_forward_backward"] = True
        except RuntimeError as error:
            errors.append(f"BF16 execution failed: {error}")
    elif not args.allow_cpu:
        errors.append("CUDA is not available")
    try:
        result["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    result["errors"] = errors
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
