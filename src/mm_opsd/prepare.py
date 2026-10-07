import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from mm_opsd.io import write_json, write_jsonl


SOURCES = {
    "geometry3k": "hiyouga/geometry3k",
    "mathvista": "AI4Math/MathVista",
    "mathvision": "MathLLMs/MathVision",
}
DEFAULT_REVISIONS = {
    "geometry3k": "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
    "mathvista": "2b6ad69445fbb5695c9b165475e8decdbeb97747",
    "mathvision": "2837ddb3f13abaf6b3997c12d80753e5470bd46a",
}


def split_files(files, split):
    selected = [name for name in files if name.startswith(f"data/{split}-")
                and name.endswith(".parquet")]
    if not selected:
        raise ValueError(f"No parquet files found for split {split}")
    return selected


def image_signature(image):
    image = image.convert("RGB")
    exact = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    difference = sum(
        int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8) for x in range(8)
    )
    return {"sha256": exact, "dhash": f"{difference:016x}", "aspect": image.width / image.height}


def signatures_overlap(left, right, max_distance=2):
    if left["sha256"] == right["sha256"]:
        return True
    aspect_close = abs(left["aspect"] / right["aspect"] - 1) <= 0.1
    distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
    return aspect_close and distance <= max_distance


def remove_overlaps(training, held_out, max_distance=2):
    kept, removed = [], []
    for row in training:
        match = next((item for item in held_out if signatures_overlap(
            row["image_signature"], item["image_signature"], max_distance
        )), None)
        if match is None:
            kept.append(row)
        else:
            removed.append({
                "train_id": row["id"], "held_out_id": match["id"],
                "train_question": row["question"], "held_out_question": match["question"],
                "train_image": row["image"], "held_out_image": match["image"],
                "exact_image": row["image_signature"]["sha256"] == match["image_signature"]["sha256"],
            })
    return kept, removed


def convert_row(name, split, index, original, root):
    if name == "geometry3k":
        image = original["images"][0]
        question, choices = original["problem"], []
    else:
        image = original["decoded_image"]
        question = original["question"]
        choices = original.get("choices", original.get("options")) or []
    if not isinstance(image, Image.Image):
        raise TypeError(f"{name}/{split}/{index}: expected a decoded PIL image")
    answer = str(original["answer"]).strip()
    if not answer:
        raise ValueError(f"{name}/{split}/{index}: missing public answer")
    image = image.convert("RGB")
    identifier = f"{name}:{split}:{original.get('pid', original.get('id', index))}"
    image_path = Path("images") / name / split / f"{index:06d}.png"
    (root / image_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(root / image_path)
    precision = original.get("precision")
    return {
        "id": identifier, "dataset": name, "split": split,
        "question": re.sub(r"<image\d*>", "", question).strip(),
        "answer": answer, "image": image_path.as_posix(), "choices": list(choices),
        "question_type": original.get("question_type", "multi_choice" if choices else "free_form"),
        "answer_type": original.get("answer_type", "math"),
        "precision": int(precision) if precision is not None else None,
        "metadata": original.get("metadata", {"subject": original.get("subject")}),
        "image_signature": image_signature(image),
    }


def main():
    parser = argparse.ArgumentParser(description="Download, standardize and decontaminate visual math data")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--include-mathvision", action="store_true")
    parser.add_argument("--geometry-revision", default=DEFAULT_REVISIONS["geometry3k"])
    parser.add_argument("--mathvista-revision", default=DEFAULT_REVISIONS["mathvista"])
    parser.add_argument("--mathvision-revision", default=DEFAULT_REVISIONS["mathvision"])
    parser.add_argument("--dhash-distance", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.dhash_distance <= 64:
        raise ValueError("dhash-distance must be between 0 and 64")
    root = Path(args.output)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.include_mathvision and "mathvision" not in manifest["sources"]:
            raise SystemExit("Use a new --output directory to add MathVision; training exclusions must be rebuilt.")
        for name, source in manifest["sources"].items():
            requested = getattr(args, f"{name.replace('3k', '')}_revision")
            if requested != source["revision"]:
                raise SystemExit(f"Dataset revision changed for {name}; use a new --output directory.")
        if args.dhash_distance != manifest["overlap_policy"]["dhash_distance"]:
            raise SystemExit("Overlap settings changed; use a new --output directory.")
        for name in manifest["counts"]:
            if not (root / f"{name}.jsonl").exists():
                raise FileNotFoundError(f"Prepared split is missing: {name}")
        print(f"Prepared data already exists: {manifest_path}")
        return

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm

    api = HfApi()
    root.mkdir(parents=True, exist_ok=True)
    names = ["geometry3k", "mathvista"] + (["mathvision"] if args.include_mathvision else [])
    sources, rows = {}, {}
    for name in names:
        requested = getattr(args, f"{name.replace('3k', '')}_revision", None)
        info = api.dataset_info(SOURCES[name], revision=requested)
        revision = info.sha
        splits = ["train", "validation", "test"] if name == "geometry3k" else ["testmini"]
        sources[name] = {"repository": SOURCES[name], "revision": revision}
        for split in splits:
            files = split_files([item.rfilename for item in info.siblings], split)
            downloaded = [hf_hub_download(SOURCES[name], filename, repo_type="dataset", revision=revision)
                          for filename in files]
            raw = load_dataset("parquet", data_files={split: downloaded}, split=split)
            rows[(name, split)] = [
                convert_row(name, split, i, item, root)
                for i, item in enumerate(tqdm(raw, desc=f"{name}/{split}"))
            ]

    held_out = [item for (name, split), items in rows.items()
                if not (name == "geometry3k" and split == "train") for item in items]
    original_count = len(rows[("geometry3k", "train")])
    clean, removed = remove_overlaps(rows[("geometry3k", "train")], held_out, args.dhash_distance)
    if not clean:
        raise RuntimeError("Overlap filtering removed all training data; inspect image signatures.")
    rows[("geometry3k", "train")] = clean
    for (name, split), items in rows.items():
        write_jsonl(root / f"{name}_{split}.jsonl", items)
    write_jsonl(root / "removed_overlaps.jsonl", removed)
    manifest = {
        "schema_version": 1, "sources": sources,
        "counts": {f"{name}_{split}": len(items) for (name, split), items in rows.items()},
        "original_training_count": original_count, "removed_training_count": len(removed),
        "overlap_policy": {"exact_rgb": True, "dhash_distance": args.dhash_distance,
                           "aspect_relative_tolerance": 0.1, "image_first_conservative": True},
        "supervision": "final_answer_only",
        "limitations": "Image hashing is a conservative overlap screen, not proof of complete decontamination.",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
