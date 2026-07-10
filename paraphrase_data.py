"""Generate LLM query paraphrases and semantic checks for authority data.

The model only sees the query conditions during paraphrase generation. Authority
rules and labels are copied from the base row after generation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.make_base_authority_data import GENERAL_AUTHORITY, TOOL_AUTHORITY
from src.make_paraphrase_data import (
    read_jsonl,
    render_user_policy,
)


DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "base"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "paraphrase"
DEFAULT_DATASET_NAMES = (GENERAL_AUTHORITY, TOOL_AUTHORITY)
DEFAULT_SPLIT_NAMES = ("train", "test")
DEFAULT_MODEL = os.environ.get("OPENAI_PARAPHRASE_MODEL", "gpt-5.4-mini")
DEFAULT_PARAPHRASE_VERSION = "v2"
DEFAULT_CACHE_PATH = (
    REPO_ROOT
    / "data"
    / "paraphrase_cache"
    / f"{DEFAULT_PARAPHRASE_VERSION}.jsonl"
)

MODEL_PRICING_USD_PER_1M_TOKENS = {
    "input": 0.75,
    "cached_input": 0.075,
    "output": 4.50,
}

PARAPHRASE_INSTRUCTIONS = """\
Rewrite structured query conditions into one natural English action request.

Rules:
- Use only the query conditions in the input.
- Preserve every condition exactly.
- Do not add policy, permission, priority, user, label, or authority-rule facts.
- Do not mention datasets, configs, records, rows, filters, or searching.
- Phrase it as an action happening under those conditions.
- If a tool is provided, make the request about using that exact tool.
- Return JSON only: {"natural_query": "..."}.

Example:
{"query_conditions":{"day":"saturday","time":"10:00-10:59"}}
-> {"natural_query":"Action on Saturday from 10:00 to 10:59."}
"""

EVALUATION_INSTRUCTIONS = """\
Score whether the natural query has exactly the same meaning as the structured
query conditions.

Return only one character: 0 or 1.

Return 1 only if all structured conditions are preserved and no extra condition
is introduced. Dataset/config names, records, rows, filters, or searching are
extra meaning and must be scored 0. Otherwise return 0.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM paraphrases for query conditions and evaluate whether "
            "the paraphrase preserves the structured condition meaning."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=("generate-cache", "build-output", "generate-and-build"),
        default="generate-cache",
        help=(
            "Default: generate-cache. generate-cache only creates/updates "
            "query paraphrase cache; build-output replaces Query blocks from "
            "cache; generate-and-build does both."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASET_NAMES),
        help="Dataset directories/configs to process.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLIT_NAMES),
        help="Split names to process.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Model for semantic scoring. Defaults to --model.",
    )
    parser.add_argument("--paraphrase-version", default=DEFAULT_PARAPHRASE_VERSION)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=(
            "JSONL cache keyed by input_text + paraphrase_version. "
            f"Default: {DEFAULT_CACHE_PATH}"
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached paraphrases and regenerate them.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Deprecated alias for --mode build-output.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--eval-max-output-tokens", type=int, default=16)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first generation/evaluation payload without API calls.",
    )
    return parser.parse_args()


def is_tool_authority(dataset_name: str) -> bool:
    return dataset_name.startswith("ToolAuthority")


def user_display_name(user: str) -> str:
    return f"User {user}"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def query_payload(base_row: dict[str, Any]) -> dict[str, Any]:
    query = base_row["Query"]
    return query_payload_from_query(query)


def query_payload_from_query(query: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query_conditions": query.get("attributes", {}),
    }
    tool = query.get("tool")
    if tool:
        payload["tool"] = tool
    return payload


def input_text_for_row(base_row: dict[str, Any]) -> str:
    return compact_json(query_payload(base_row))


def cache_key(input_text: str, paraphrase_version: str) -> tuple[str, str]:
    return input_text, paraphrase_version


def load_paraphrase_cache(cache_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                key = cache_key(row["input_text"], row["paraphrase_version"])
            except KeyError as exc:
                raise ValueError(
                    f"Invalid cache row at {cache_path}:{line_number}"
                ) from exc
            cache[key] = row
    return cache


def append_paraphrase_cache_row(cache_path: Path, row: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_output_row(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_existing_output_rows(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    return read_jsonl(output_path)


def output_row_input_text(row: dict[str, Any]) -> str | None:
    metadata = row.get("meta_data") or row.get("metadata") or {}
    if "ParaphraseInputText" in metadata:
        return metadata["ParaphraseInputText"]
    query = metadata.get("query") or metadata.get("AttributeBase")
    if query:
        return compact_json(query_payload_from_query(query))
    return None


def output_row_paraphrase_version(row: dict[str, Any]) -> str | None:
    metadata = row.get("meta_data") or row.get("metadata") or {}
    return metadata.get("paraphrase_version") or metadata.get("ParaphraseVersion")


def is_slim_output_row(row: dict[str, Any]) -> bool:
    metadata = row.get("meta_data") or {}
    return set(row) == {"text", "label", "meta_data"} and {
        "query",
        "paraphrase_version",
        "query_meaning_score",
    }.issubset(metadata)


def validate_existing_output_rows(
    *,
    output_path: Path,
    existing_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    paraphrase_version: str,
) -> None:
    if len(existing_rows) > len(base_rows):
        raise ValueError(
            f"{output_path} has {len(existing_rows)} rows, but this run only "
            f"needs {len(base_rows)} rows. Use a different --output-dir or "
            "remove the stale output file before restarting."
        )

    for row_idx, existing_row in enumerate(existing_rows):
        if not is_slim_output_row(existing_row):
            raise ValueError(
                f"{output_path} row {row_idx} uses an older non-slim schema. "
                "Use a fresh --output-dir or remove the stale output file before "
                "building push-ready data."
            )
        expected_input_text = input_text_for_row(base_rows[row_idx])
        found_input_text = output_row_input_text(existing_row)
        found_version = output_row_paraphrase_version(existing_row)
        if (
            found_input_text != expected_input_text
            or found_version != paraphrase_version
        ):
            raise ValueError(
                f"{output_path} row {row_idx} does not match the requested "
                "base row/paraphrase version. Use a different --output-dir or "
                "remove the stale output file before restarting."
            )


def progress_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    total: int,
    initial: int,
    desc: str,
):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return rows
    return tqdm(rows, total=total, initial=initial, desc=desc, unit="row")


def render_text_with_natural_query(
    base_row: dict[str, Any],
    natural_query: str,
) -> str:
    authority_setting = base_row["AuthoritySetting"]
    dataset_name = str(authority_setting.get("dataset", ""))
    priority = base_row["metadata"]["priority"]
    tool = authority_setting.get("tool")

    if is_tool_authority(dataset_name):
        scope = f"Task: decide whether the requested use of the {tool} tool is authorized."
    elif tool:
        scope = f"Task: decide whether {tool} is authorized."
    else:
        scope = "Task: decide whether the requested action is authorized."

    instruction = (
        "Multiple users provide allow/disallow rules for the requested action. "
        "If their rules disagree, follow the priority order."
    )
    priority_text = " > ".join(user_display_name(user) for user in priority)
    user_text = "\n\n".join(
        render_user_policy(user_setting)
        for user_setting in authority_setting["users"]
    )
    query_header = "Query conditions" if is_tool_authority(dataset_name) else "Query"
    return (
        f"{scope}\n"
        f"{instruction}\n"
        f"Priority: {priority_text}\n\n"
        f"{user_text}\n\n"
        f"{query_header}:\n{natural_query}"
    )


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_json_fence(text)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got: {type(value).__name__}")
    return value


def parse_binary_score(text: str) -> int:
    stripped = strip_json_fence(text).strip()
    if stripped in {"0", "1"}:
        return int(stripped)

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\b[01]\b", stripped)
        if not match:
            raise
        return int(match.group(0))

    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    if isinstance(value, dict) and "score" in value:
        score = int(value["score"])
        if score in {0, 1}:
            return score
    raise ValueError(f"Evaluation score must be 0 or 1, got {value!r}.")


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "input_tokens_details", None)
    cached_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
    }


def estimate_cost_usd(usage: dict[str, int]) -> float:
    input_tokens = usage.get("input_tokens", 0)
    cached_tokens = usage.get("cached_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    uncached_tokens = max(input_tokens - cached_tokens, 0)
    return (
        uncached_tokens * MODEL_PRICING_USD_PER_1M_TOKENS["input"]
        + cached_tokens * MODEL_PRICING_USD_PER_1M_TOKENS["cached_input"]
        + output_tokens * MODEL_PRICING_USD_PER_1M_TOKENS["output"]
    ) / 1_000_000


def create_response(
    client: Any,
    *,
    model: str,
    instructions: str,
    payload: dict[str, Any],
    max_output_tokens: int,
) -> tuple[str, dict[str, int], float]:
    max_output_tokens = max(max_output_tokens, 16)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=compact_json(payload),
        max_output_tokens=max_output_tokens,
        store=False,
    )
    usage = usage_to_dict(getattr(response, "usage", None))
    return response_text(response), usage, estimate_cost_usd(usage)


def generate_natural_query(
    client: Any,
    *,
    model: str,
    base_row: dict[str, Any],
    max_output_tokens: int,
) -> tuple[str, dict[str, int], float, str]:
    payload = query_payload(base_row)
    raw_text, usage, cost = create_response(
        client,
        model=model,
        instructions=PARAPHRASE_INSTRUCTIONS,
        payload=payload,
        max_output_tokens=max_output_tokens,
    )
    parsed = parse_json_object(raw_text)
    natural_query = str(parsed["natural_query"]).strip()
    if not natural_query:
        raise ValueError("Model returned an empty natural_query.")
    return natural_query, usage, cost, raw_text


def evaluate_natural_query(
    client: Any,
    *,
    model: str,
    base_row: dict[str, Any],
    natural_query: str,
    max_output_tokens: int,
) -> tuple[int, dict[str, int], float, str]:
    payload = {
        "structured_query": query_payload(base_row),
        "natural_query": natural_query,
    }
    raw_text, usage, cost = create_response(
        client,
        model=model,
        instructions=EVALUATION_INSTRUCTIONS,
        payload=payload,
        max_output_tokens=max_output_tokens,
    )
    return parse_binary_score(raw_text), usage, cost, raw_text


def cache_row_evaluation_score(cache_row: dict[str, Any]) -> int:
    if "evaluation_score" in cache_row:
        score = int(cache_row["evaluation_score"])
    else:
        evaluation = cache_row.get("evaluation", {})
        if isinstance(evaluation, dict):
            score = int(evaluation["score"])
        else:
            score = int(evaluation)
    if score not in {0, 1}:
        raise ValueError(f"Evaluation score must be 0 or 1, got {score!r}.")
    return score


def make_cache_row(
    *,
    input_text: str,
    paraphrase_version: str,
    generated_text: str,
    evaluation_score: int,
) -> dict[str, Any]:
    return {
        "input_text": input_text,
        "paraphrase_version": paraphrase_version,
        "generated_text": generated_text,
        "evaluation_score": evaluation_score,
    }


def cache_row_generated_text(cache_row: dict[str, Any]) -> str:
    return cache_row.get("generated_text") or cache_row["natural_query"]


def get_or_create_cache_row(
    *,
    client: Any | None,
    cache: dict[tuple[str, str], dict[str, Any]],
    cache_path: Path,
    refresh_cache: bool,
    cache_only: bool,
    base_row: dict[str, Any],
    model: str,
    eval_model: str,
    paraphrase_version: str,
    max_output_tokens: int,
    eval_max_output_tokens: int,
) -> tuple[dict[str, Any], bool, float]:
    input_text = input_text_for_row(base_row)
    key = cache_key(input_text, paraphrase_version)
    if not refresh_cache and key in cache:
        return cache[key], True, 0.0
    if cache_only:
        raise KeyError(
            "Missing paraphrase cache entry for "
            f"paraphrase_version={paraphrase_version!r}, input_text={input_text!r}"
        )
    if client is None:
        client = load_client()

    natural_query, _, gen_cost, _ = generate_natural_query(
        client,
        model=model,
        base_row=base_row,
        max_output_tokens=max_output_tokens,
    )
    evaluation_score, _, eval_cost, _ = evaluate_natural_query(
        client,
        model=eval_model,
        base_row=base_row,
        natural_query=natural_query,
        max_output_tokens=eval_max_output_tokens,
    )
    row = make_cache_row(
        input_text=input_text,
        paraphrase_version=paraphrase_version,
        generated_text=natural_query,
        evaluation_score=evaluation_score,
    )
    cache[key] = row
    append_paraphrase_cache_row(cache_path, row)
    return row, False, gen_cost + eval_cost


def make_output_row(
    base_row: dict[str, Any],
    *,
    natural_query: str,
    evaluation_score: int,
    paraphrase_version: str,
) -> dict[str, Any]:
    return {
        "text": render_text_with_natural_query(base_row, natural_query),
        "label": base_row["Label"],
        "meta_data": {
            "query": base_row["Query"],
            "paraphrase_version": paraphrase_version,
            "query_meaning_score": evaluation_score,
        },
    }


def selected_rows(rows: list[dict[str, Any]], *, offset: int, limit: int | None) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("--offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    sliced = rows[offset:]
    return sliced if limit is None else sliced[:limit]


def load_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The `openai` package is required. Install it in this environment first."
        ) from exc
    return OpenAI()


def print_dry_run(
    input_path: Path,
    row: dict[str, Any],
    *,
    paraphrase_version: str,
) -> None:
    payload = query_payload(row)
    input_text = input_text_for_row(row)
    print(f"[dry-run] input: {input_path}")
    print(f"[dry-run] paraphrase version: {paraphrase_version}")
    print("[dry-run] cache key:")
    print(
        json.dumps(
            {"input_text": input_text, "paraphrase_version": paraphrase_version},
            indent=2,
            ensure_ascii=False,
        )
    )
    print("[dry-run] paraphrase instructions:")
    print(PARAPHRASE_INSTRUCTIONS)
    print("[dry-run] paraphrase input:")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    print("[dry-run] evaluation instructions:")
    print(EVALUATION_INSTRUCTIONS)
    print("[dry-run] evaluation input shape:")
    print(
        json.dumps(
            {
                "structured_query": payload,
                "natural_query": "<model-generated natural query>",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def process_split(
    *,
    client: Any | None,
    cache: dict[tuple[str, str], dict[str, Any]],
    cache_path: Path,
    input_path: Path,
    output_path: Path,
    model: str,
    eval_model: str,
    paraphrase_version: str,
    refresh_cache: bool,
    mode: str,
    offset: int,
    limit: int | None,
    max_output_tokens: int,
    eval_max_output_tokens: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    base_rows = selected_rows(read_jsonl(input_path), offset=offset, limit=limit)
    build_output = mode in {"build-output", "generate-and-build"}
    generate_missing = mode in {"generate-cache", "generate-and-build"}

    existing_output_rows = []
    if build_output:
        existing_output_rows = read_existing_output_rows(output_path)
        validate_existing_output_rows(
            output_path=output_path,
            existing_rows=existing_output_rows,
            base_rows=base_rows,
            paraphrase_version=paraphrase_version,
        )

    output_rows = list(existing_output_rows)
    start_idx = len(existing_output_rows) if build_output else 0
    remaining_rows = list(enumerate(base_rows[start_idx:], start=start_idx))
    split_cost = 0.0
    cache_hits = 0
    cache_misses = 0
    score_1 = 0
    score_0 = 0
    if generate_missing:
        for _, base_row in remaining_rows:
            key = cache_key(input_text_for_row(base_row), paraphrase_version)
            if refresh_cache or key not in cache:
                client = client or load_client()
                break

    for row_idx, base_row in progress_rows(
        remaining_rows,
        total=len(base_rows),
        initial=start_idx,
        desc=input_path.parent.name + "/" + input_path.stem,
    ):
        cache_row, cache_hit, row_cost = get_or_create_cache_row(
            client=client,
            cache=cache,
            cache_path=cache_path,
            refresh_cache=refresh_cache,
            cache_only=not generate_missing,
            base_row=base_row,
            model=model,
            eval_model=eval_model,
            paraphrase_version=paraphrase_version,
            max_output_tokens=max_output_tokens,
            eval_max_output_tokens=eval_max_output_tokens,
        )
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
        split_cost += row_cost
        evaluation_score = cache_row_evaluation_score(cache_row)
        if evaluation_score == 1:
            score_1 += 1
        else:
            score_0 += 1
        if build_output:
            output_row = make_output_row(
                base_row,
                natural_query=cache_row_generated_text(cache_row),
                evaluation_score=evaluation_score,
                paraphrase_version=cache_row.get(
                    "paraphrase_version",
                    paraphrase_version,
                ),
            )
            append_output_row(output_path, output_row)
            output_rows.append(output_row)
        if sleep_seconds and not cache_hit:
            time.sleep(sleep_seconds)

    return {
        "rows": len(output_rows) if build_output else len(base_rows),
        "processed": len(remaining_rows),
        "skipped_existing": start_idx,
        "output_written": len(remaining_rows) if build_output else 0,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "estimated_cost_usd": round(split_cost, 8),
        "score_1": (
            sum(row["meta_data"]["query_meaning_score"] == 1 for row in output_rows)
            if build_output
            else score_1
        ),
        "score_0": (
            sum(row["meta_data"]["query_meaning_score"] == 0 for row in output_rows)
            if build_output
            else score_0
        ),
    }


def main() -> None:
    args = parse_args()
    eval_model = args.eval_model or args.model
    counts: dict[str, dict[str, Any]] = {}
    if args.cache_only:
        args.mode = "build-output"
    if args.mode == "build-output" and args.refresh_cache:
        raise ValueError("--mode build-output and --refresh-cache cannot be used together.")
    if (
        args.cache_path == DEFAULT_CACHE_PATH
        and args.paraphrase_version != DEFAULT_PARAPHRASE_VERSION
    ):
        args.cache_path = (
            REPO_ROOT
            / "data"
            / "paraphrase_cache"
            / f"{args.paraphrase_version}.jsonl"
        )

    if args.dry_run:
        for dataset_name in args.datasets:
            for split_name in args.splits:
                input_path = args.input_dir / dataset_name / f"{split_name}.jsonl"
                if input_path.exists():
                    rows = selected_rows(read_jsonl(input_path), offset=args.offset, limit=1)
                    if rows:
                        print_dry_run(
                            input_path,
                            rows[0],
                            paraphrase_version=args.paraphrase_version,
                        )
                        return
        raise FileNotFoundError("No input rows found for dry run.")

    cache = load_paraphrase_cache(args.cache_path)
    client = None
    for dataset_name in args.datasets:
        counts[dataset_name] = {}
        for split_name in args.splits:
            input_path = args.input_dir / dataset_name / f"{split_name}.jsonl"
            if not input_path.exists():
                continue
            output_path = args.output_dir / dataset_name / f"{split_name}.jsonl"
            print(f"[paraphrase] {dataset_name}/{split_name}: {input_path}")
            counts[dataset_name][split_name] = process_split(
                client=client,
                cache=cache,
                cache_path=args.cache_path,
                input_path=input_path,
                output_path=output_path,
                model=args.model,
                eval_model=eval_model,
                paraphrase_version=args.paraphrase_version,
                refresh_cache=args.refresh_cache,
                mode=args.mode,
                offset=args.offset,
                limit=args.limit,
                max_output_tokens=args.max_output_tokens,
                eval_max_output_tokens=args.eval_max_output_tokens,
                sleep_seconds=args.sleep_seconds,
            )

    print(json.dumps(counts, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
