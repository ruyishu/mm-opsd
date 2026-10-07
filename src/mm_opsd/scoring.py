import ast
import os
import re
from decimal import Decimal, InvalidOperation


def boxed_answers(text):
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[start:index].strip())
                    break
    return values


def extract_answer(text):
    boxes = boxed_answers(text)
    if boxes:
        return boxes[-1]
    tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if tags:
        return tags[-1].strip()
    marked = re.findall(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and len(lines[0]) <= 128:
        return lines[0]
    return None


def normalized_text(value):
    value = str(value).strip().strip("$")
    value = re.sub(r"\\(?:left|right)", "", value)
    value = value.replace("\\,", "").replace("\\!", "")
    return re.sub(r"\s+", "", value).rstrip(".")


def choice_index(answer, choices):
    answer = str(answer).strip()
    match = re.fullmatch(r"\(?([A-Z])\)?(?:[.):]\s*.*)?", answer)
    if match and ord(match[1]) - 65 < len(choices):
        return ord(match[1]) - 65
    matches = [i for i, choice in enumerate(choices) if normalized_text(answer) == normalized_text(choice)]
    return matches[0] if matches else None


def numeric_value(text):
    text = normalized_text(text).replace(",", "")
    try:
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?", text):
            numerator, denominator = text.split("/")
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(text)
    except (InvalidOperation, ZeroDivisionError):
        return None


def math_equal(prediction, answer):
    if normalized_text(prediction) == normalized_text(answer):
        return True
    left, right = numeric_value(prediction), numeric_value(answer)
    if left is not None and right is not None and left.is_finite() and right.is_finite():
        return abs(left - right) <= Decimal("1e-8") * max(Decimal(1), abs(right))
    from math_verify import LatexExtractionConfig, parse, verify
    # Math-Verify 0.8 uses an unpicklable nested process target on Windows.
    timeout = 2 if os.name == "posix" else None
    try:
        expected = parse(f"${answer}$", extraction_config=[LatexExtractionConfig()], parsing_timeout=timeout)
        actual = parse(f"${prediction}$", extraction_config=[LatexExtractionConfig()], parsing_timeout=timeout)
        return bool(expected and actual and verify(expected, actual, timeout_seconds=timeout))
    except (ValueError, TypeError, TimeoutError, RuntimeError, OverflowError):
        return False


def score_answer(text, row):
    prediction = extract_answer(text)
    result = {"prediction": prediction, "parse_failed": prediction is None, "correct": False}
    if prediction is None:
        return result
    choices = row.get("choices") or []
    if choices:
        actual = choice_index(prediction, choices)
        expected = choice_index(row["answer"], choices)
        result["parse_failed"] = actual is None
        result["correct"] = (actual is not None and expected is not None
                             and normalized_text(choices[actual]) == normalized_text(choices[expected]))
    elif row.get("answer_type") in {"float", "integer"}:
        actual, expected = numeric_value(prediction), numeric_value(row["answer"])
        if actual is None or not actual.is_finite():
            result["parse_failed"] = True
        elif expected is not None and expected.is_finite():
            precision = row.get("precision")
            if row.get("answer_type") == "integer":
                result["correct"] = actual == expected and actual == actual.to_integral_value()
            elif precision is not None:
                result["correct"] = round(float(actual), precision) == round(float(expected), precision)
            else:
                result["correct"] = actual == expected
    elif row.get("answer_type") == "list":
        try:
            result["correct"] = ast.literal_eval(prediction) == ast.literal_eval(row["answer"])
        except (ValueError, SyntaxError):
            result["parse_failed"] = True
    else:
        result["correct"] = math_equal(prediction, row["answer"])
    return result
