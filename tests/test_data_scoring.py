from PIL import Image, ImageDraw

from mm_opsd.prepare import image_signature, remove_overlaps, split_files
from mm_opsd.report import paired_statistics
from mm_opsd.scoring import extract_answer, score_answer


def test_conservative_image_overlap_removes_training_only():
    shared = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(shared).line((0, 0, 63, 63), fill="black", width=4)
    other = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(other).rectangle((10, 10, 50, 50), fill="black")
    def row(identifier, image):
        return {"id": identifier, "question": "Find x.", "image": identifier + ".png",
                "image_signature": image_signature(image)}
    training = [row("train:0", shared), row("train:1", other)]
    held_out = [row("test:0", shared)]
    kept, removed = remove_overlaps(training, held_out)
    assert [item["id"] for item in kept] == ["train:1"]
    assert removed[0]["held_out_id"] == "test:0"
    assert len(held_out) == 1


def test_answer_extraction_and_equivalence():
    assert extract_answer(r"Reasoning ... \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_answer("Unfinished\nreasoning without an answer") is None
    row = {"answer": "0.5", "choices": [], "answer_type": "math"}
    assert score_answer(r"\boxed{\frac{1}{2}}", row)["correct"]
    assert not score_answer(r"\boxed{0.6}", row)["correct"]
    assert score_answer("unfinished\nreasoning", row)["parse_failed"]


def test_choice_letter_and_value_scoring():
    row = {"answer": "145°", "choices": ["135°", "140°", "145°", "150°"]}
    assert score_answer(r"\boxed{C}", row)["correct"]
    assert score_answer("145°", row)["correct"]
    assert not score_answer(r"\boxed{D}", row)["correct"]
    assert score_answer("Z", row)["parse_failed"]


def test_mathvista_float_precision_and_integer():
    row = {"answer": "1.2", "choices": [], "answer_type": "float", "precision": 1}
    assert score_answer(r"\boxed{1.23}", row)["correct"]
    assert not score_answer(r"\boxed{1.26}", row)["correct"]
    integer = {"answer": "1", "choices": [], "answer_type": "integer"}
    assert not score_answer("1.2", integer)["correct"]


def test_mathvista_duplicate_correct_choices():
    row = {"answer": "18", "choices": ["9", "12", "18", "18"], "answer_type": "text"}
    for answer in ["18", "C", "D"]:
        result = score_answer("\\boxed{" + answer + "}", row)
        assert result["correct"] and not result["parse_failed"]
    assert not score_answer(r"\boxed{B}", row)["correct"]


def test_paired_report_and_id_mismatch():
    import pytest
    left = [{"id": str(i), "correct": False} for i in range(4)]
    right = [{"id": str(i), "correct": i < 2} for i in range(4)]
    result = paired_statistics(left, right, repetitions=100)
    assert result["accuracy_delta"] == 0.5
    assert result["improved_questions"] == 2
    assert result["mcnemar_exact_p_value"] == 0.5
    with pytest.raises(ValueError):
        paired_statistics(left, right[::-1])


def test_download_only_requested_evaluation_split():
    files = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet",
             "data/testmini-00000-of-00001.parquet", "README.md"]
    assert split_files(files, "testmini") == ["data/testmini-00000-of-00001.parquet"]
