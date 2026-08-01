"""Offline quality evaluation for ``run_eval`` generation artifacts.

The evaluator consumes JSONL artifacts rather than loading datasets or models.  Each
record needs a dataset name (or ``--dataset``) and a generated text field such as
``output``, ``prediction``, ``completion``, or ``answer``.  References are read from
common fields (``reference``, ``label``, ``target``, and dataset-specific fields).

HumanEval candidates are *only* executed in a fresh child process.  On Linux the
child installs CPU, address-space, file-size, process-count, and core-dump limits;
the parent additionally enforces a wall-clock timeout for every problem.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# This source is deliberately executed with a separate Python interpreter.  Keep
# candidate execution out of the evaluator process, including in error paths.
_HUMANEVAL_WORKER = r'''
import contextlib
import io
import json
import math
import os
import sys

request = json.loads(sys.stdin.read())
limits_applied = False
if sys.platform.startswith("linux"):
    import resource
    # Limits are defense in depth, not a complete sandbox.  The parent provides
    # the wall-clock timeout, while RLIMIT_CPU covers CPU consumption in the child.
    cpu_seconds = max(1, int(math.ceil(request["timeout_seconds"])))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (request["memory_bytes"], request["memory_bytes"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (request["file_bytes"], request["file_bytes"]))
    resource.setrlimit(resource.RLIMIT_NPROC, (request["process_limit"], request["process_limit"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    limits_applied = True

result = {"passed": False, "status": "error", "resource_limits_applied": limits_applied}
# Suppress ordinary candidate output so the parent receives exactly one JSON object.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        namespace = {"__name__": "__main__"}
        exec(compile(request["source"], "<humaneval-candidate>", "exec"), namespace)
        exec(compile(request["test"], "<humaneval-test>", "exec"), namespace)
        check = namespace.get("check")
        if callable(check):
            check(namespace[request["entry_point"]])
        result.update(passed=True, status="passed")
    except AssertionError:
        result["status"] = "failed"
    except BaseException as exc:
        result.update(status="error", error=type(exc).__name__)
print(json.dumps(result, sort_keys=True))
'''


_DATASET_ALIASES = {
    "humaneval": "humaneval",
    "human_eval": "humaneval",
    "openai_humaneval": "humaneval",
    "math": "math500",
    "math500": "math500",
    "math_500": "math500",
    "nq": "nq",
    "nq_open": "nq",
    "nq_qa": "nq",
    "nq_rag": "nq",
    "natural_questions": "nq",
    "wmt23": "wmt23",
    "wmt_23": "wmt23",
    "wmt": "wmt23",
    "alpaca": "alpaca",
    "alpaca_eval": "alpaca",
    "mtbench": "mtbench",
    "mt_bench": "mtbench",
}


def canonical_dataset(name: str) -> str:
    """Return the supported canonical dataset name or raise ``ValueError``."""
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    try:
        return _DATASET_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_DATASET_ALIASES.values())))
        raise ValueError(f"unsupported dataset {name!r}; expected one of {supported}") from exc


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def prediction_from_record(record: Mapping[str, Any]) -> Optional[str]:
    """Extract generated text from common ``run_eval`` artifact field names."""
    value = _first(record, ("prediction", "output", "generated_text", "completion", "answer"))
    if isinstance(value, Mapping):
        value = _first(value, ("text", "content", "output", "answer"))
    return _as_text(value)


def reference_from_record(record: Mapping[str, Any]) -> Any:
    return _first(record, ("reference", "references", "label", "target", "gold"))


def _humaneval_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return HumanEval execution metadata in the current ``run_eval`` schema."""
    evaluation = record.get("evaluation")
    if isinstance(evaluation, Mapping):
        return evaluation
    reference = reference_from_record(record)
    if isinstance(reference, Mapping):
        return reference
    return record


def _conversation_prompt(record: Mapping[str, Any]) -> Optional[str]:
    conversation = record.get("conversation")
    if not isinstance(conversation, Sequence) or isinstance(conversation, (str, bytes)):
        return None
    for message in reversed(conversation):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _as_text(message.get("content"))
    return None


def _strip_code_fences(text: str, entry_point: str) -> str:
    fences = list(re.finditer(r"```[^\n`]*\n(.*?)```", text, re.DOTALL))
    if not fences:
        return text.strip()
    candidates = [match.group(1).strip() for match in fences]
    function_pattern = re.compile(rf"(?:^|\n)(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
    for candidate in candidates:
        if function_pattern.search(candidate):
            return candidate
    return candidates[0]


def extract_humaneval_code(text: str, entry_point: str) -> str:
    """Extract fenced Python, a complete function, or a raw completion.

    The raw completion case intentionally preserves indentation: HumanEval prompts
    commonly end at a function body's indentation level.
    """
    candidate = _strip_code_fences(text, entry_point)
    if _has_entry_point_definition(candidate, entry_point):
        # Keep imports and helpers when the complete fenced/code-only response is
        # syntactically valid.  Otherwise discard surrounding natural language.
        try:
            ast.parse(candidate)
        except SyntaxError:
            pass
        else:
            return candidate.strip()
    match = re.search(
        rf"(?:^|\n)((?:async\s+)?def\s+{re.escape(entry_point)}\s*\()", candidate
    )
    if match:
        return candidate[match.start(1) :].strip()
    return candidate


def _has_entry_point_definition(code: str, entry_point: str) -> bool:
    return bool(
        re.search(
            rf"(?:^|\n)(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code
        )
    )


def _human_eval_source(prompt: str, prediction: str, entry_point: str) -> str:
    code = extract_humaneval_code(prediction, entry_point)
    if _has_entry_point_definition(code, entry_point):
        return code
    if not prompt:
        raise ValueError("a completion without a full entry-point function needs prompt")
    return prompt + code


def evaluate_humaneval_case(
    prompt: str,
    prediction: str,
    test: str,
    entry_point: str,
    *,
    timeout_seconds: float = 3.0,
    memory_bytes: int = 1 << 30,
    file_bytes: int = 10 << 20,
    process_limit: int = 32,
) -> Dict[str, Any]:
    """Evaluate one HumanEval candidate in an isolated, resource-limited process."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    source = _human_eval_source(prompt, prediction, entry_point)
    request = {
        "source": source,
        "test": test,
        "entry_point": entry_point,
        "timeout_seconds": timeout_seconds,
        "memory_bytes": memory_bytes,
        "file_bytes": file_bytes,
        "process_limit": process_limit,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _HUMANEVAL_WORKER],
            input=json.dumps(request),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "status": "timeout", "resource_limits_applied": sys.platform.startswith("linux")}

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "status": "worker_error",
            "returncode": completed.returncode,
            "resource_limits_applied": sys.platform.startswith("linux"),
        }
    if completed.returncode != 0 and result.get("status") == "passed":
        result.update(passed=False, status="worker_error", returncode=completed.returncode)
    return result


def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return None


def _extract_balanced_parentheses(text: str) -> Optional[str]:
    if not text.startswith("("):
        return None
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return None


def _trim_math_tail(text: str) -> str:
    """Remove presentation wrappers and terminal prose punctuation."""
    result = text.strip()
    result = re.sub(r"^(?:[:=]|is|are|equals?|gives?|becomes?)\s*", "", result, flags=re.IGNORECASE)
    result = re.sub(r"(?:\$|\\\)|\]|[.?!;:])\s*$", "", result).strip()
    result = re.sub(r"^\$|^\\\(", "", result).strip()
    return result


def _numeric_surface(text: str) -> Optional[str]:
    """Return the last syntactic numeric surface without evaluating it."""
    number = r"[+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?"
    radical = r"(?:\s*(?:√\s*(?:\{[^{}]+\}|[A-Za-z0-9]+)|\\sqrt\s*(?:\{[^{}]+\}|[A-Za-z0-9]+)))?"
    degree = r"(?:\s*(?:°|\\circ|\^\{?\\circ\}?|degrees?))?"
    latex_fraction = r"[+-]?\\(?:d?frac)\{[^{}]+\}\{[^{}]+\}"
    matches = list(re.finditer(rf"(?:{latex_fraction}|{number}{radical}{degree})", text))
    return matches[-1].group(0).strip() if matches else None


def _math_surface_from_tail(text: str) -> Optional[str]:
    """Extract an answer-shaped surface from a conclusion clause."""
    candidate = _trim_math_tail(text)
    if not candidate:
        return None

    parenthesized = _extract_balanced_parentheses(candidate)
    if parenthesized is not None:
        return parenthesized

    # Preserve a complete LaTex fraction before discarding explanatory words.
    fraction = re.match(r"[+-]?\\(?:d?frac)\{[^{}]+\}\{[^{}]+\}", candidate)
    if fraction:
        return fraction.group(0)

    numeric = _numeric_surface(candidate)
    if numeric is not None:
        return numeric

    # Multiple-choice/name answers are textual surfaces, not symbolic expressions.
    word = re.match(r"([A-Z][A-Za-z'\-]*)\b", candidate)
    return word.group(1) if word else None


def _conclusion_surface(text: str) -> Optional[str]:
    """Use the last equality or copula in a final conclusion clause."""
    equalities = list(re.finditer(r"(?<![<>=])=(?!=)\s*([^\n]+)", text))
    if equalities:
        surface = _math_surface_from_tail(equalities[-1].group(1))
        if surface is not None:
            return surface

    copulas = list(
        re.finditer(
            r"\b(?:is|are|was|were|equals?|has|have|had|gives?|becomes?)\b\s+([^\n]+)",
            text,
            re.IGNORECASE,
        )
    )
    if copulas:
        surface = _math_surface_from_tail(copulas[-1].group(1))
        if surface is not None:
            return surface
    return None


def _last_conclusion_segment(text: str) -> Tuple[str, bool]:
    """Return the final non-empty line/sentence and whether it is terminated."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "", False
    line = lines[-1]
    terminated = bool(re.search(r"[.?!]\s*$", line))
    sentences = [part.strip() for part in re.split(r"(?<=[.?!])\s+", line) if part.strip()]
    return sentences[-1] if sentences else line, terminated


def extract_math_answer(text: str) -> Optional[str]:
    """Extract a final MATH answer with conservative DeepSeek-Math-style priority.

    Priority is: last ``\\boxed{}``, explicit final/answer/conclusion wording,
    final equality or conclusion clause, then a final-sentence numeric fallback.
    This selects a surface form only; it never algebraically evaluates expressions.
    """
    boxed = list(re.finditer(r"\\boxed\s*", text))
    for match in reversed(boxed):
        value = _extract_balanced_braces(text, match.end())
        if value is not None:
            return value

    explicit = list(
        re.finditer(
            r"\b(?:final(?:\s+(?:answer|result|value))?|answer|conclusion|therefore|thus|hence)\b\s*(.*)",
            text,
            re.IGNORECASE,
        )
    )
    for match in reversed(explicit):
        surface = _conclusion_surface(match.group(1)) or _math_surface_from_tail(match.group(1))
        if surface is not None:
            return surface

    segment, terminated = _last_conclusion_segment(text)
    surface = _conclusion_surface(segment)
    if surface is not None:
        return surface

    # A bare answer (including tuples, radicals, and option names) is valid.  Do
    # not use an arbitrary earlier derivation: numeric fallback is final-only and
    # requires a complete sentence when it lacks a conclusion cue.
    bare = _math_surface_from_tail(segment)
    if bare is not None and _trim_math_tail(segment) == bare:
        return bare
    if terminated:
        return _numeric_surface(segment)
    return None


def normalize_math_answer(value: Any) -> Optional[str]:
    """Normalize answer surfaces for exact match; this is not a symbolic judge."""
    text = _as_text(value)
    if text is None:
        return None
    extracted = extract_math_answer(text)
    if extracted is None:
        return None
    result = extracted.strip().lower()
    result = re.sub(r"^\$+|\$+$", "", result)
    result = re.sub(r"^\\\((.*)\\\)$", r"\1", result)
    result = re.sub(r"\\(?:left|right|,|!|;|:| )", "", result)
    result = re.sub(r"\\text\{([^{}]*)\}", r"\1", result)
    # Syntactic canonicalization only: fractions, radicals, pi, and degrees.
    result = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", result)
    result = re.sub(r"√\s*(?:\{([^{}]+)\}|([a-z0-9]+))", lambda m: f"sqrt({m.group(1) or m.group(2)})", result)
    result = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", result)
    result = re.sub(r"\\sqrt\s*([a-z0-9]+)", r"sqrt(\1)", result)
    result = result.replace("π", "pi").replace("\\pi", "pi")
    result = re.sub(r"(?:\^\{?\\circ\}?|\\circ|°|\bdegrees?\b)", "deg", result)
    result = re.sub(r"\b(?:units?|meters?|metres?|centimeters?|centimetres?|cm|mm|feet|ft|inches?|miles?)\b", "", result)
    result = re.sub(r"\s+", "", result)
    if re.fullmatch(r"[+-]?\d+", result):
        try:
            result = str(int(result))
        except ValueError:
            pass
    return result or None


def normalize_nq(text: str) -> str:
    """NQ/SQuAD normalization for exact match and token F1."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(char for char in text if not _is_punctuation(char))
    return " ".join(text.split())


def _is_punctuation(char: str) -> bool:
    # Avoid a dependency while retaining the SQuAD definition of ASCII punctuation.
    return char in r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_nq(prediction).split()
    reference_tokens = normalize_nq(reference).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(reference_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _nq_references(record: Mapping[str, Any]) -> List[str]:
    reference = reference_from_record(record)
    if reference is None:
        reference = _first(record, ("answers", "answer"))
    if isinstance(reference, Mapping):
        reference = _first(reference, ("text", "answers", "answer", "value"))
    if isinstance(reference, str):
        return [reference]
    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        return [text for value in reference if (text := _as_text(value)) is not None]
    return []


def _wmt_references(record: Mapping[str, Any]) -> List[str]:
    reference = reference_from_record(record)
    if reference is None:
        reference = _first(record, ("translation", "answer"))
    if isinstance(reference, Mapping):
        reference = _first(reference, ("text", "translation", "target", "answer"))
    if isinstance(reference, str):
        return [reference]
    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        return [text for item in reference if (text := _as_text(item)) is not None]
    return []


def _closest_reference_length(candidate_length: int, references: Sequence[Sequence[str]]) -> int:
    lengths = [len(reference) for reference in references]
    return min(lengths, key=lambda length: (abs(length - candidate_length), length))


def _ngram_counts(tokens: Sequence[str], order: int) -> collections.Counter[Tuple[str, ...]]:
    return collections.Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def self_contained_corpus_bleu4(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """Corpus BLEU-4, whitespace-tokenized and unsmoothed, on a 0--100 scale."""
    matches = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    candidate_length = 0
    reference_length = 0
    for prediction, item_references in zip(predictions, references):
        candidate_tokens = prediction.split()
        reference_tokens = [reference.split() for reference in item_references]
        if not reference_tokens:
            continue
        candidate_length += len(candidate_tokens)
        reference_length += _closest_reference_length(len(candidate_tokens), reference_tokens)
        for order in range(1, 5):
            candidate_counts = _ngram_counts(candidate_tokens, order)
            totals[order - 1] += sum(candidate_counts.values())
            max_reference_counts: collections.Counter[Tuple[str, ...]] = collections.Counter()
            for reference in reference_tokens:
                for ngram, count in _ngram_counts(reference, order).items():
                    max_reference_counts[ngram] = max(max_reference_counts[ngram], count)
            matches[order - 1] += sum(
                min(count, max_reference_counts[ngram]) for ngram, count in candidate_counts.items()
            )
    if not candidate_length or any(total == 0 or match == 0 for match, total in zip(matches, totals)):
        return 0.0
    log_precision = sum(math.log(match / total) for match, total in zip(matches, totals)) / 4
    brevity_penalty = 1.0 if candidate_length > reference_length else math.exp(1 - reference_length / candidate_length)
    return 100 * brevity_penalty * math.exp(log_precision)


def _evaluate_humaneval(records: Sequence[Mapping[str, Any]], timeout_seconds: float) -> Dict[str, Any]:
    passed = failed = timed_out = errors = evaluated = 0
    linux_limits = sys.platform.startswith("linux")
    for record in records:
        prediction = prediction_from_record(record)
        metadata = _humaneval_metadata(record)
        prompt = (
            _as_text(_first(metadata, ("prompt", "question", "input")))
            or _conversation_prompt(record)
            or _as_text(record.get("prompt"))
            or ""
        )
        test = _as_text(_first(metadata, ("test", "tests"))) or _as_text(record.get("test"))
        entry_point = _as_text(_first(metadata, ("entry_point", "entrypoint"))) or _as_text(record.get("entry_point"))
        if prediction is None or test is None or not entry_point:
            errors += 1
            continue
        try:
            result = evaluate_humaneval_case(prompt, prediction, test, entry_point, timeout_seconds=timeout_seconds)
        except ValueError:
            errors += 1
            continue
        evaluated += 1
        linux_limits = linux_limits and bool(result.get("resource_limits_applied"))
        if result.get("passed"):
            passed += 1
        elif result.get("status") == "timeout":
            timed_out += 1
        elif result.get("status") == "failed":
            failed += 1
        else:
            errors += 1
    return {
        "metric": "pass@1",
        "pass_at_1": passed / evaluated if evaluated else None,
        "passed": passed,
        "failed": failed,
        "timeouts": timed_out,
        "errors": errors,
        "evaluated": evaluated,
        "resource_limits": "linux_rlimits" if sys.platform.startswith("linux") else "unavailable_non_linux",
        "all_evaluated_children_limited": linux_limits if evaluated else None,
    }


def _evaluate_math500(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct = evaluated = extraction_failures = 0
    for record in records:
        raw_reference = reference_from_record(record)
        if raw_reference is None:
            raw_reference = record.get("answer")
        # A reference is the inclusion criterion.  Never silently drop a scored
        # example just because the generation omitted a final/boxed answer.
        if raw_reference is None:
            continue
        evaluated += 1
        prediction = normalize_math_answer(prediction_from_record(record))
        reference = normalize_math_answer(raw_reference)
        if prediction is None:
            extraction_failures += 1
            continue
        # Dataset references should be normalized answers.  If a malformed
        # reference appears, retain it in the denominator rather than inflating
        # accuracy; it is not a prediction-extraction failure.
        if reference is not None and prediction == reference:
            correct += 1
    return {
        "metric": "exact_match_normalized_non_symbolic",
        "symbolic_judge": False,
        "accuracy": correct / evaluated if evaluated else None,
        "correct": correct,
        "evaluated": evaluated,
        "extraction_failures": extraction_failures,
    }


def _evaluate_nq(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    em_total = f1_total = 0.0
    evaluated = 0
    for record in records:
        prediction = prediction_from_record(record)
        references = _nq_references(record)
        if prediction is None or not references:
            continue
        evaluated += 1
        normalized_prediction = normalize_nq(prediction)
        em_total += max(float(normalized_prediction == normalize_nq(reference)) for reference in references)
        f1_total += max(_token_f1(prediction, reference) for reference in references)
    return {
        "metric": "normalized_exact_match_and_token_f1",
        "exact_match": em_total / evaluated if evaluated else None,
        "token_f1": f1_total / evaluated if evaluated else None,
        "evaluated": evaluated,
    }


def _evaluate_wmt23(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[List[str]] = []
    for record in records:
        prediction = prediction_from_record(record)
        item_references = _wmt_references(record)
        if prediction is not None and item_references:
            predictions.append(prediction)
            references.append(item_references)
    try:
        import sacrebleu  # type: ignore
    except ImportError:
        return {
            "metric": "self_contained_corpus_bleu4_whitespace_unsmoothed",
            "bleu": self_contained_corpus_bleu4(predictions, references),
            "evaluated": len(predictions),
        }
    transposed = [list(items) for items in zip(*references)] if references else [[]]
    score = sacrebleu.corpus_bleu(predictions, transposed)
    return {
        "metric": "sacrebleu_corpus_bleu",
        "bleu": score.score,
        "signature": str(score),
        "evaluated": len(predictions),
    }


def _record_temperature(record: Mapping[str, Any]) -> Any:
    """Read the generation temperature from current and older artifact layouts."""
    temperature = _first(record, ("temperature", "temp"))
    if temperature is not None:
        return temperature
    for key in ("generation_config", "generation_kwargs", "gen_kwargs"):
        config = record.get(key)
        if isinstance(config, Mapping) and config.get("temperature") is not None:
            return config["temperature"]
    return None


def _temperature_key(temperature: Any) -> str:
    """Create stable JSON keys while grouping numeric and string temperatures."""
    if temperature is None:
        return "unspecified"
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        return format(temperature, "g")
    text = str(temperature).strip()
    try:
        return format(float(text), "g")
    except ValueError:
        return text or "unspecified"


def _evaluate_dataset(
    name: str, records: Sequence[Mapping[str, Any]], timeout_seconds: float
) -> Dict[str, Any]:
    if name == "humaneval":
        return _evaluate_humaneval(records, timeout_seconds)
    if name == "math500":
        return _evaluate_math500(records)
    if name == "nq":
        return _evaluate_nq(records)
    if name == "wmt23":
        return _evaluate_wmt23(records)
    return {
        "metric": "requires_judge",
        "requires_judge": True,
        "accuracy": None,
        "status": "N/A",
        "evaluated": 0,
        "records": len(records),
    }


def evaluate_records(
    records: Iterable[Mapping[str, Any]], *, dataset: Optional[str] = None, timeout_seconds: float = 3.0
) -> Dict[str, Any]:
    """Return one JSON-serializable summary grouped by dataset and temperature.

    A dataset with one temperature retains the original direct metric layout under
    ``datasets[dataset]``.  Mixed-temperature artifacts use
    ``datasets[dataset]["temperatures"][temperature]`` so T=0 and T=1 cannot be
    accidentally merged into one quality score.
    """
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    forced_dataset = canonical_dataset(dataset) if dataset else None
    total = 0
    for record in records:
        total += 1
        if not isinstance(record, Mapping):
            raise ValueError("every JSONL line must be a JSON object")
        name = forced_dataset or _first(record, ("dataset", "benchmark", "task"))
        if not isinstance(name, str):
            raise ValueError("records need a dataset field, or pass --dataset")
        dataset_name = canonical_dataset(name)
        temperature = _record_temperature(record)
        key = (dataset_name, _temperature_key(temperature))
        bucket = grouped.setdefault(key, {"temperature": temperature, "records": []})
        bucket["records"].append(record)

    by_dataset: Dict[str, List[Tuple[str, Dict[str, Any]]]] = collections.defaultdict(list)
    for (name, temperature_key), bucket in grouped.items():
        by_dataset[name].append((temperature_key, bucket))

    summaries: Dict[str, Any] = {}
    for name, temperature_buckets in sorted(by_dataset.items()):
        temperature_summaries: Dict[str, Dict[str, Any]] = {}
        for temperature_key, bucket in sorted(temperature_buckets, key=lambda item: item[0]):
            result = _evaluate_dataset(name, bucket["records"], timeout_seconds)
            result["temperature"] = bucket["temperature"]
            temperature_summaries[temperature_key] = result
        # Preserve the pre-temperature public layout for conventional single-T
        # artifacts, while still recording which temperature produced the score.
        if len(temperature_summaries) == 1:
            summaries[name] = next(iter(temperature_summaries.values()))
        else:
            summaries[name] = {"temperatures": temperature_summaries}
    return {"total_records": total, "datasets": summaries}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


FORMAL_HF_ALGORITHMS = ("baseline", "eagle3", "specblock")
FORMAL_HF_DATASETS = ("humaneval", "math500", "alpaca", "nq_open", "mtbench", "wmt23")
FORMAL_HF_TEMPERATURES = (0.0, 1.0)
FORMAL_HF_BATCHES = (16, 32)
FORMAL_HF_CONFIGS = {"baseline": "B,0,0,0", "eagle3": "B,7,10,60", "specblock": "B,2,10,90"}
FORMAL_HF_MAX_NEW_TOKENS = 1024
FORMAL_HF_EXPECTED_SAMPLES = {"humaneval": 164, "math500": 200, "alpaca": 200, "nq_open": 200, "mtbench": 80, "wmt23": 200}


def _formal_temperature(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid temperature {value!r}") from exc
    if value not in FORMAL_HF_TEMPERATURES:
        raise ValueError(f"unexpected formal temperature {value!r}")
    return value


def _formal_key(record: Mapping[str, Any]) -> Tuple[int, str, str, float, str]:
    return (
        int(record["batch_size"]),
        str(record["algorithm"]),
        str(record["benchmark"]),
        _formal_temperature(record["temperature"]),
        str(record["config"]),
    )


def _status_exit_code(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("exit_status="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def validate_hf_formal_root(root: Path, *, write_result: bool = True) -> Dict[str, Any]:
    """Fail closed unless an immutable BS16/32 formal matrix is fully complete."""
    root = root.resolve()
    errors: List[str] = []
    manifest_path = root / "formal_matrix_manifest.json"
    manifest: Dict[str, Any] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"missing or invalid manifest: {exc}")
    if manifest:
        if manifest.get("framework") != "hf":
            errors.append("manifest framework must be hf")
        if manifest.get("run_root") != str(root):
            errors.append("manifest run_root does not match immutable root")
        if tuple(manifest.get("batches", ())) != FORMAL_HF_BATCHES:
            errors.append("formal HF batches must be exactly [16, 32]")
        if tuple(manifest.get("algorithms", ())) != FORMAL_HF_ALGORITHMS:
            errors.append("formal HF algorithms do not match contract")
        if tuple(manifest.get("datasets", ())) != FORMAL_HF_DATASETS:
            errors.append("formal HF datasets do not match contract")
        try:
            manifest_temperatures = tuple(float(v) for v in manifest.get("temperatures", ()))
        except (TypeError, ValueError):
            manifest_temperatures = ()
        if manifest_temperatures != FORMAL_HF_TEMPERATURES:
            errors.append("formal HF temperatures must be exactly [0.0, 1.0]")
        if manifest.get("configs") != FORMAL_HF_CONFIGS:
            errors.append("formal HF configs do not match frozen contract")
        if manifest.get("expected_examples_by_dataset") != FORMAL_HF_EXPECTED_SAMPLES:
            errors.append("formal HF expected_examples_by_dataset does not match contract")
        if manifest.get("max_new_tokens") != FORMAL_HF_MAX_NEW_TOKENS:
            errors.append("formal HF max_new_tokens must be 1024")
        if manifest.get("expected_records") != 72:
            errors.append("manifest expected_records must be 72")
    run_id = manifest.get("run_id") if manifest else None
    configs = FORMAL_HF_CONFIGS
    expected = set()
    for batch in FORMAL_HF_BATCHES:
        for algorithm in FORMAL_HF_ALGORITHMS:
            config = configs.get(algorithm, "").replace("B", str(batch))
            if not config:
                errors.append(f"manifest lacks config for {algorithm}")
                continue
            for dataset in FORMAL_HF_DATASETS:
                for temperature in FORMAL_HF_TEMPERATURES:
                    expected.add((batch, algorithm, dataset, temperature, config))

    observed: Dict[Tuple[int, str, str, float, str], Mapping[str, Any]] = {}
    for batch in FORMAL_HF_BATCHES:
        batch_dir = root / f"b{batch}"
        protocol = batch_dir / "protocol.json"
        try:
            protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
            if protocol_data.get("run_id") != run_id or protocol_data.get("run_root") != str(root):
                errors.append(f"{protocol}: run identity mismatch")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"missing or invalid batch protocol {protocol}: {exc}")
        try:
            worker_identity = json.loads((batch_dir / "worker_identity.json").read_text(encoding="utf-8"))
            if (worker_identity.get("run_id") != run_id or not worker_identity.get("python_repo_path")
                    or worker_identity.get("source_identity") != manifest.get("source_identity")):
                errors.append(f"worker source identity mismatch for B{batch}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"missing or invalid worker identity for B{batch}: {exc}")
        for algorithm in FORMAL_HF_ALGORITHMS:
            status = batch_dir / f"{algorithm}.status"
            if _status_exit_code(status) != 0:
                errors.append(f"failed or missing status artifact: {status}")
            summary_path = batch_dir / f"{algorithm}.jsonl"
            try:
                rows = read_jsonl(summary_path)
            except (OSError, ValueError) as exc:
                errors.append(f"missing or invalid summary artifact {summary_path}: {exc}")
                continue
            for row in rows:
                try:
                    key = _formal_key(row)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"invalid summary key in {summary_path}: {exc}")
                    continue
                if key in observed:
                    errors.append(f"duplicate summary key: {key}")
                    continue
                observed[key] = row
                required = ("requested_samples", "measured_samples", "max_new_tokens", "node", "visible_gpu", "provenance", "run_id", "run_root", "sample_artifact")
                missing = [name for name in required if row.get(name) in (None, "")]
                if missing:
                    errors.append(f"summary {key} missing required provenance: {missing}")
                if row.get("run_id") != run_id or row.get("run_root") != str(root):
                    errors.append(f"summary {key} run identity mismatch")
                if row.get("provenance") != manifest:
                    errors.append(f"summary {key} provenance does not equal immutable manifest")
                if row.get("max_new_tokens") != FORMAL_HF_MAX_NEW_TOKENS:
                    errors.append(f"summary {key} max_new_tokens violates formal contract")
                expected_samples = FORMAL_HF_EXPECTED_SAMPLES[key[2]]
                if row.get("requested_samples") != expected_samples or row.get("measured_samples") != expected_samples:
                    errors.append(f"summary {key} requested/measured must both equal {expected_samples}")
                if row.get("failed_requests") != 0:
                    errors.append(f"summary {key} has failed requests")
                sample_name = row.get("sample_artifact")
                sample_path = (batch_dir / str(sample_name)).resolve()
                try:
                    sample_path.relative_to(root)
                except ValueError:
                    errors.append(f"sample artifact escapes immutable root: {sample_name!r}")
                    continue
                try:
                    samples = read_jsonl(sample_path)
                except (OSError, ValueError) as exc:
                    errors.append(f"missing or invalid sample artifact {sample_path}: {exc}")
                    continue
                partition_samples = []
                for sample in samples:
                    try:
                        sample_temperature = _formal_temperature(sample.get("temperature"))
                    except ValueError:
                        sample_temperature = None
                    sample_key = (sample.get("batch_size"), sample.get("algorithm"), sample.get("dataset"), sample_temperature, sample.get("config_key"))
                    if sample_key not in expected:
                        errors.append(f"unexpected sample partition in {sample_path}: {sample_key}")
                        continue
                    if (sample.get("algorithm") == algorithm and sample.get("dataset") == key[2]
                            and sample_temperature == key[3] and sample.get("config_key") == key[4]
                            and sample.get("batch_size") == batch):
                        partition_samples.append(sample)
                if len(partition_samples) != expected_samples:
                    errors.append(f"sample partition {key} in {sample_path} has {len(partition_samples)}, expected {expected_samples}")
                sample_ids = set()
                for sample in partition_samples:
                    identity = sample.get("id")
                    if identity in sample_ids:
                        errors.append(f"duplicate sample id {identity!r} in {sample_path} for {key}")
                    sample_ids.add(identity)
                    if (sample.get("run_id") != run_id or sample.get("run_root") != str(root)
                            or sample.get("requested_samples") != expected_samples or sample.get("provenance") != manifest):
                        errors.append(f"sample provenance mismatch in {sample_path}")
                        break
    missing_keys = expected - set(observed)
    extra_keys = set(observed) - expected
    if missing_keys:
        if all(key[3] == 0.0 for key in observed) and observed:
            errors.append("partial T0 root cannot be presented as formal matrix")
        errors.append(f"missing summary keys: {len(missing_keys)}")
    if extra_keys:
        errors.append(f"unexpected summary keys: {len(extra_keys)}")
    if len(observed) != 72:
        errors.append(f"observed summary records={len(observed)}, expected=72")
    failed_marker = root / "validation.failed.json"
    completion = root / "completion.json"
    failed_artifacts = [path for path in root.rglob("*.failed*") if path.is_file()]
    if failed_artifacts:
        errors.append("failed artifact exists; root is permanently failed: " + ", ".join(str(path) for path in failed_artifacts))
    result = {"status": "complete" if not errors else "failed", "run_id": run_id,
              "run_root": str(root), "expected_records": 72,
              "observed_records": len(observed), "errors": errors}
    if write_result:
        if errors:
            completion.unlink(missing_ok=True)
            failed_marker.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            if failed_marker.exists():
                errors.append("failed artifact exists; root is permanently failed")
                result["status"] = "failed"
                result["errors"] = errors
                completion.unlink(missing_ok=True)
            else:
                completion.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate_hf_external_judges(root: Path) -> Dict[str, Any]:
    """Require immutable, complete external-judge summaries for judge-only tasks."""
    root = root.resolve()
    generation = validate_hf_formal_root(root, write_result=False)
    errors = list(generation["errors"])
    if errors:
        return {"status": "failed", "errors": errors}
    for batch in FORMAL_HF_BATCHES:
        for algorithm in FORMAL_HF_ALGORITHMS:
            config = FORMAL_HF_CONFIGS[algorithm].replace("B", str(batch))
            for dataset in ("alpaca", "mtbench"):
                expected = FORMAL_HF_EXPECTED_SAMPLES[dataset]
                for temperature in FORMAL_HF_TEMPERATURES:
                    path = root / "judges" / f"b{batch}" / algorithm / dataset / f"t{temperature:g}.judge_summary.json"
                    try:
                        judge = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        errors.append(f"missing or invalid immutable judge artifact {path}: {exc}")
                        continue
                    key = judge.get("generation_partition", {})
                    if (judge.get("status") != "complete" or judge.get("run_id") != generation.get("run_id")
                            or judge.get("run_root") != str(root) or key != {"batch_size": batch, "algorithm": algorithm, "dataset": dataset, "temperature": temperature, "config": config}
                            or judge.get("expected_examples") != expected or judge.get("evaluated_examples") != expected
                            or judge.get("missing_examples") != 0 or judge.get("failed_judgments") != 0):
                        errors.append(f"incomplete judge coverage: {path}")
                    if dataset == "mtbench" and (judge.get("expected_turns") != expected * 2 or judge.get("evaluated_turns") != expected * 2):
                        errors.append(f"incomplete MT-Bench turn coverage: {path}")
    return {"status": "complete" if not errors else "failed", "errors": errors}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate run_eval *.samples.jsonl artifacts offline.")
    parser.add_argument("samples", type=Path, nargs="?", help="Path to a *.samples.jsonl generation artifact")
    parser.add_argument("--dataset", help="Dataset name when records do not include one")
    parser.add_argument("--timeout-seconds", type=float, default=3.0, help="Per-HumanEval-task wall-clock limit")
    parser.add_argument("--validate-hf-formal-root", type=Path)
    args = parser.parse_args(argv)
    if args.validate_hf_formal_root:
        result = validate_hf_formal_root(args.validate_hf_formal_root)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "complete" else 2
    if args.samples is None:
        parser.error("samples is required unless --validate-hf-formal-root is used")
    try:
        summary = evaluate_records(read_jsonl(args.samples), dataset=args.dataset, timeout_seconds=args.timeout_seconds)
    except (OSError, ValueError) as exc:
        summary = {"error": str(exc)}
        print(json.dumps(summary, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
