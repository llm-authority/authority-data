#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/data/paraphrase_cache}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/base}"

python - "$DATA_DIR" "$CACHE_DIR" <<'PY'
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


data_dir = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
paths = sorted({*data_dir.glob("*.jsonl"), *data_dir.glob("*/*.jsonl")})
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def color(text: object, code: str) -> str:
    value = str(text)
    if not USE_COLOR:
        return value
    return f"\033[{code}m{value}\033[0m"


BOLD = "1"
DIM = "2"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
MAGENTA = "35"
CYAN = "36"
RED = "31"
HEADER = "1;36"

if not paths:
    print(f"No JSONL files found under {data_dir}")
    raise SystemExit(0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def jsonl_paths(path: Path) -> list[Path]:
    return sorted({*path.glob("*.jsonl"), *path.glob("*/*.jsonl")})


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{count / total:.1%}"


def format_counter_compact(
    counter: Counter,
    total: int,
    *,
    colors: dict[Any, str] | None = None,
) -> str:
    if not counter:
        return "n/a"
    colors = colors or {}
    items = []
    for key in sorted(counter, key=lambda value: str(value)):
        count = counter[key]
        label = color(key, colors[key]) if key in colors else str(key)
        items.append(f"{label} {count} ({pct(count, total)})")
    return " | ".join(items)


CATEGORY_ABBREVIATIONS = {
    "recipient": "rec.",
    "information_type": "inf.",
    "purpose": "purp.",
    "urgency": "urg.",
}


def abbreviate_category(category: str) -> str:
    return CATEGORY_ABBREVIATIONS.get(category, category)


def format_category_combo(categories: list[str] | tuple[str, ...]) -> str:
    return "+".join(abbreviate_category(category) for category in categories)


def numeric_summary(values: list[int]) -> str:
    if not values:
        return "n/a"
    return (
        f"avg={statistics.mean(values):.2f}, "
        f"min={min(values)}, max={max(values)}"
    )


def numeric_mean_variance(values: list[float]) -> tuple[str, str]:
    if not values:
        return "n/a", "n/a"
    mean = statistics.mean(values)
    variance = statistics.pvariance(values)
    return f"{mean:.4f}", f"{variance:.4f}"


def metric_line(name: str, value: str) -> None:
    print(f"  {color((name + ':').ljust(26), DIM)} {value}")


def section(text: str) -> None:
    width = 88
    label = f" {text} "
    line = label + "-" * max(0, width - len(label))
    print(color(line, HEADER))


def table_value(value: str, column_name: str) -> str:
    if column_name == "train" and value != "0":
        return color(value, GREEN)
    if column_name == "test" and value != "0":
        return color(value, BLUE)
    if column_name == "total":
        return color(value, BOLD)
    if column_name == "Yes":
        return color(value, GREEN)
    if column_name == "No":
        return color(value, YELLOW)
    if column_name == "conflict":
        return color(value, RED)
    return value


def print_prompt_box(text: str) -> None:
    lines = text.splitlines()
    print(color("  ----- prompt -----", DIM))
    for line in lines:
        print(f"  {line}")
    print(color("  ------------------", DIM))


def authority_setting(row: dict[str, Any]) -> dict[str, Any]:
    setting = row.get("AuthoritySetting")
    if setting:
        return setting
    metadata = row.get("meta_data") or row.get("metadata") or {}
    return metadata.get("BaseAuthoritySetting", {})


def query_attributes(row: dict[str, Any]) -> dict[str, Any]:
    query = query_object(row)
    return query.get("attributes") or {}


def query_object(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("meta_data") or row.get("metadata") or {}
    return (
        row.get("AttributeCombination")
        or row.get("Query")
        or metadata.get("query")
        or metadata.get("AttributeBase")
        or {}
    )


def query_tool(row: dict[str, Any]) -> str:
    query = query_object(row)
    tool = query.get("tool")
    if tool:
        return str(tool)
    setting = authority_setting(row)
    tool = setting.get("tool")
    return str(tool) if tool else ""


def evaluation_score(row: dict[str, Any]) -> int | None:
    metadata = row.get("meta_data") or row.get("metadata") or {}
    candidates = [
        row.get("evaluation_score"),
        row.get("query_meaning_score"),
        metadata.get("query_meaning_score"),
        metadata.get("evaluation_score"),
    ]
    evaluation = row.get("evaluation") or metadata.get("evaluation")
    if isinstance(evaluation, dict):
        candidates.append(evaluation.get("score"))
    elif evaluation is not None:
        candidates.append(evaluation)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            score = int(candidate)
        except (TypeError, ValueError):
            continue
        if score in {0, 1}:
            return score
    return None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_input_text(row: dict[str, Any]) -> str:
    query = query_object(row)
    payload: dict[str, Any] = {
        "query_conditions": query.get("attributes", {}),
    }
    tool = query.get("tool")
    if tool:
        payload["tool"] = tool
    return compact_json(payload)


def row_stats(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("meta_data") or row.get("metadata") or {}
    setting = authority_setting(row)
    users = setting.get("users") or []
    attrs = query_attributes(row)
    priority = metadata.get("priority") or []
    rules_per_user = [len(user.get("rules") or []) for user in users]
    total_rules = sum(rules_per_user)

    source = metadata.get("source") or {}
    target_user = source.get("target_user", "")
    top_priority_user = priority[0] if priority else ""
    top_authority = ""
    for user in users:
        if user.get("user") == top_priority_user:
            top_authority = user.get("authority", "")
            break

    return {
        "text": row.get("text", ""),
        "label": row.get("label") or row.get("Label", ""),
        "is_conflict": bool(metadata.get("is_conflict", False)),
        "user_count": len(users),
        "priority_depth": len(priority),
        "top_priority_user": top_priority_user,
        "top_authority": top_authority,
        "target_user": target_user,
        "query_k": sum(value is not None for value in attrs.values()),
        "tool": query_tool(row),
        "category_combo": tuple(metadata.get("categories") or sorted(attrs)),
        "categories": format_category_combo(metadata.get("categories") or sorted(attrs)),
        "total_rules": total_rules,
        "rules_per_user_min": min(rules_per_user) if rules_per_user else 0,
        "rules_per_user_max": max(rules_per_user) if rules_per_user else 0,
        "eval_score": evaluation_score(row),
        "label_matches_top": (row.get("label") or row.get("Label", "")).lower()
        == top_authority,
        "target_matches_top": not target_user or target_user == top_priority_user,
    }


def load_eval_cache(cache_paths: list[Path]) -> dict[tuple[str, str], int]:
    cache: dict[tuple[str, str], int] = {}
    for path in cache_paths:
        for row in read_jsonl(path):
            input_text = row.get("input_text")
            version = row.get("paraphrase_version")
            score = evaluation_score(row)
            if input_text is None or version is None or score is None:
                continue
            cache[(str(input_text), str(version))] = score
    return cache


def paraphrase_dataset_name(dataset: str, version: str) -> str:
    if version == "v1":
        return dataset
    return f"{dataset}-Paraphrase"


def print_table(rows: list[list[str]], *, title: str) -> None:
    if not rows:
        return
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    print(title)
    headers = rows[0]
    for row_index, row in enumerate(rows):
        rendered = []
        for index, value in enumerate(row):
            cell = value.ljust(widths[index])
            if row_index == 0:
                cell = color(cell, BOLD)
            else:
                cell = table_value(cell, headers[index])
            rendered.append(cell)
        print("  ".join(rendered))
        if row_index == 0:
            print(color("  ".join("-" * width for width in widths), DIM))
    print()


def cache_group_name(path: Path, root: Path) -> tuple[str, str]:
    dataset = path.parent.name if path.parent != root else root.name
    return dataset, path.stem


def print_eval_cache_summary(cache_paths: list[Path], *, title: str) -> None:
    if not cache_paths:
        return

    rows = [
        [
            "dataset",
            "split",
            "rows",
            "eval count",
            "eval 1",
            "eval 0",
            "eval avg",
            "eval var",
        ]
    ]
    all_scores: list[int] = []
    for path in cache_paths:
        dataset, split = cache_group_name(path, cache_dir)
        cache_rows = read_jsonl(path)
        scores = [
            score
            for score in (evaluation_score(row) for row in cache_rows)
            if score is not None
        ]
        all_scores.extend(scores)
        eval_avg, eval_var = numeric_mean_variance(scores)
        rows.append(
            [
                dataset,
                split,
                str(len(cache_rows)),
                str(len(scores)),
                str(sum(score == 1 for score in scores)),
                str(sum(score == 0 for score in scores)),
                eval_avg,
                eval_var,
            ]
        )

    if len(cache_paths) > 1:
        eval_avg, eval_var = numeric_mean_variance(all_scores)
        rows.append(
            [
                "overall",
                "-",
                str(sum(int(row[2]) for row in rows[1:])),
                str(len(all_scores)),
                str(sum(score == 1 for score in all_scores)),
                str(sum(score == 0 for score in all_scores)),
                eval_avg,
                eval_var,
            ]
        )

    section(title)
    print_table(rows, title=color("Evaluation scores from paraphrase cache", DIM))


def print_eval_by_split_summary(
    data_paths: list[Path],
    cache_paths: list[Path],
    *,
    title: str,
) -> None:
    if not data_paths or not cache_paths:
        return

    cache = load_eval_cache(cache_paths)
    versions = sorted({version for _, version in cache})
    if not versions:
        return

    rows = [
        [
            "dataset",
            "split",
            "base rows",
            "eval count",
            "eval 1",
            "eval 0",
            "missing",
            "V2 rows",
            "eval avg",
            "eval var",
        ]
    ]
    for version in versions:
        for path in data_paths:
            dataset = path.parent.name if path.parent != data_dir else data_dir.name
            split = path.stem
            data_rows = read_jsonl(path)
            scores = []
            missing = 0
            for row in data_rows:
                key = (cache_input_text(row), version)
                score = cache.get(key)
                if score is None:
                    missing += 1
                    continue
                scores.append(score)
            if not scores and missing == 0:
                continue

            eval_avg, eval_var = numeric_mean_variance(scores)
            rows.append(
                [
                    paraphrase_dataset_name(dataset, version),
                    split,
                    str(len(data_rows)),
                    str(len(scores)),
                    str(sum(score == 1 for score in scores)),
                    str(sum(score == 0 for score in scores)),
                    str(missing),
                    str(sum(score == 1 for score in scores)),
                    eval_avg,
                    eval_var,
                ]
            )

    if len(rows) == 1:
        return

    section(title)
    print_table(
        rows,
        title=color(
            "Evaluation scores applied to each V1 split; V2 rows keep eval=1 only",
            DIM,
        ),
    )


def choose_prompt_example(
    dataset: str,
    all_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[str, dict[str, Any] | None]:
    preferred_user_count = 2 if dataset.startswith("ToolAuthority") else None

    if preferred_user_count is not None:
        for split in ("train", "test"):
            for row in all_groups.get((dataset, split), []):
                if row["user_count"] == preferred_user_count:
                    return split, row

    for split in ("train", "test"):
        stats = all_groups.get((dataset, split), [])
        if stats:
            return split, stats[0]

    return "", None


all_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
for path in paths:
    dataset = path.parent.name if path.parent != data_dir else data_dir.name
    split = path.stem
    all_groups[(dataset, split)] = [row_stats(row) for row in read_jsonl(path)]


section("Authority Data Stats")
print(f"{color('data_dir:', DIM)} {data_dir}")
print(f"{color('cache_dir:', DIM)} {cache_dir}")
print()

compact_rows = [
    [
        "dataset",
        "split",
        "rows",
        "Yes",
        "No",
        "conflict",
        "users",
        "rules avg",
        "eval avg",
        "eval var",
    ]
]
for (dataset, split), stats in sorted(all_groups.items()):
    total = len(stats)
    labels = Counter(row["label"] for row in stats)
    conflict = Counter(row["is_conflict"] for row in stats)
    users = Counter(row["user_count"] for row in stats)
    rules = [row["total_rules"] for row in stats]
    eval_scores = [
        row["eval_score"]
        for row in stats
        if row["eval_score"] is not None
    ]
    eval_avg, eval_var = numeric_mean_variance(eval_scores)
    compact_rows.append(
        [
            dataset,
            split,
            str(total),
            f"{labels['Yes']} ({pct(labels['Yes'], total)})",
            f"{labels['No']} ({pct(labels['No'], total)})",
            f"{conflict[True]} ({pct(conflict[True], total)})",
            ", ".join(f"{key}:{users[key]}" for key in sorted(users)),
            f"{statistics.mean(rules):.2f}" if rules else "n/a",
            eval_avg,
            eval_var,
        ]
    )
section("Split Summary")
print_table(
    compact_rows,
    title=color("Rows, labels, conflicts, users, rule counts, and eval scores", DIM),
)

section("Category Combinations")
for dataset in sorted({dataset for dataset, _ in all_groups}):
    category_rows = [["category_combo", "train", "test", "total"]]
    split_counters = {
        split: Counter(row["categories"] for row in all_groups.get((dataset, split), []))
        for split in ("train", "test")
    }
    combos = sorted(set(split_counters["train"]) | set(split_counters["test"]))
    for combo in combos:
        train_count = split_counters["train"][combo]
        test_count = split_counters["test"][combo]
        category_rows.append(
            [
                combo,
                str(train_count),
                str(test_count),
                str(train_count + test_count),
            ]
        )
    print_table(
        category_rows,
        title=color(f"{dataset} category combinations by split", DIM),
    )

tool_datasets = sorted(
    {
        dataset
        for dataset, _ in all_groups
        if any(
            row["tool"]
            for split in ("train", "test")
            for row in all_groups.get((dataset, split), [])
        )
    }
)
for dataset in tool_datasets:
    split_counters = {
        split: Counter(
            row["tool"]
            for row in all_groups.get((dataset, split), [])
            if row["tool"]
        )
        for split in ("train", "test")
    }
    tools = sorted(set(split_counters["train"]) | set(split_counters["test"]))
    if not tools:
        continue
    tool_rows = [["tool", "train", "test", "total"]]
    for tool in tools:
        train_count = split_counters["train"][tool]
        test_count = split_counters["test"][tool]
        tool_rows.append(
            [
                tool,
                str(train_count),
                str(test_count),
                str(train_count + test_count),
            ]
        )
    section("Tool Split")
    print_table(tool_rows, title=color(f"{dataset} tools by split", DIM))

section("Prompt Examples")
for dataset in sorted({dataset for dataset, _ in all_groups}):
    example_split, example = choose_prompt_example(dataset, all_groups)
    if example is None:
        continue

    print(color(f"{dataset}/{example_split}", BOLD))
    print_prompt_box(example["text"])
    metric_line("label", color(example["label"], GREEN if example["label"] == "Yes" else YELLOW))
    metric_line("user_count", color(example["user_count"], CYAN))
    print()

combined = [row for stats in all_groups.values() for row in stats]
combined_total = len(combined)
if len(all_groups) > 1:
    section("Overall")
    metric_line("rows", color(combined_total, BOLD))
    metric_line(
        "labels",
        format_counter_compact(
            Counter(row["label"] for row in combined),
            combined_total,
            colors={"Yes": GREEN, "No": YELLOW},
        ),
    )
    metric_line(
        "conflict",
        format_counter_compact(
            Counter(row["is_conflict"] for row in combined),
            combined_total,
            colors={True: RED, False: GREEN},
        ),
    )
    metric_line(
        "user_count",
        format_counter_compact(
            Counter(row["user_count"] for row in combined),
            combined_total,
            colors={1: CYAN, 2: BLUE, 3: MAGENTA, 4: YELLOW},
        ),
    )
    metric_line("total_rules", numeric_summary([row["total_rules"] for row in combined]))
    metric_line("query_k", numeric_summary([row["query_k"] for row in combined]))
    eval_scores = [
        row["eval_score"]
        for row in combined
        if row["eval_score"] is not None
    ]
    eval_avg, eval_var = numeric_mean_variance(eval_scores)
    metric_line("eval_score_count", str(len(eval_scores)))
    metric_line("eval_score_avg", eval_avg)
    metric_line("eval_score_var", eval_var)

cache_paths = jsonl_paths(cache_dir) if cache_dir.exists() else []
print_eval_by_split_summary(paths, cache_paths, title="Eval Scores By Split")
print_eval_cache_summary(cache_paths, title="Eval Cache Summary")
PY
