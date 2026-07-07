#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/paraphrase}"

python - "$DATA_DIR" <<'PY'
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


data_dir = Path(sys.argv[1])
paths = sorted(data_dir.glob("*/*.jsonl"))
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


def numeric_summary(values: list[int]) -> str:
    if not values:
        return "n/a"
    return (
        f"avg={statistics.mean(values):.2f}, "
        f"min={min(values)}, max={max(values)}"
    )


def metric_line(name: str, value: str) -> None:
    print(f"  {color((name + ':').ljust(26), DIM)} {value}")


def authority_setting(row: dict[str, Any]) -> dict[str, Any]:
    setting = row.get("AuthoritySetting")
    if setting:
        return setting
    return row.get("metadata", {}).get("BaseAuthoritySetting", {})


def query_attributes(row: dict[str, Any]) -> dict[str, Any]:
    query = row.get("AttributeCombination") or row.get("Query") or {}
    return query.get("attributes") or {}


def row_stats(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
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
        "label": row.get("Label", ""),
        "is_conflict": bool(metadata.get("is_conflict", False)),
        "user_count": len(users),
        "priority_depth": len(priority),
        "top_priority_user": top_priority_user,
        "top_authority": top_authority,
        "target_user": target_user,
        "query_k": sum(value is not None for value in attrs.values()),
        "categories": "+".join(metadata.get("categories") or sorted(attrs)),
        "total_rules": total_rules,
        "rules_per_user_min": min(rules_per_user) if rules_per_user else 0,
        "rules_per_user_max": max(rules_per_user) if rules_per_user else 0,
        "label_matches_top": row.get("Label", "").lower() == top_authority,
        "target_matches_top": not target_user or target_user == top_priority_user,
    }


def print_table(rows: list[list[str]], *, title: str) -> None:
    if not rows:
        return
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    print(title)
    for row_index, row in enumerate(rows):
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))
    print()


all_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
for path in paths:
    dataset = path.parent.name
    split = path.stem
    all_groups[(dataset, split)] = [row_stats(row) for row in read_jsonl(path)]


def title(text: str) -> None:
    print(color(text, BOLD))


print(f"{color('data_dir:', DIM)} {data_dir}")
print()

compact_rows = [["dataset", "split", "rows", "Yes", "No", "conflict", "users", "rules avg"]]
for (dataset, split), stats in sorted(all_groups.items()):
    total = len(stats)
    labels = Counter(row["label"] for row in stats)
    conflict = Counter(row["is_conflict"] for row in stats)
    users = Counter(row["user_count"] for row in stats)
    rules = [row["total_rules"] for row in stats]
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
        ]
    )
print_table(compact_rows, title=color("Split Summary", BOLD))

for (dataset, split), stats in sorted(all_groups.items()):
    total = len(stats)
    title(f"{dataset}/{split}")
    metric_line(
        "labels",
        format_counter_compact(
            Counter(row["label"] for row in stats),
            total,
            colors={"Yes": GREEN, "No": YELLOW},
        ),
    )
    metric_line(
        "conflict",
        format_counter_compact(
            Counter(row["is_conflict"] for row in stats),
            total,
            colors={True: RED, False: GREEN},
        ),
    )
    metric_line(
        "user_count",
        format_counter_compact(
            Counter(row["user_count"] for row in stats),
            total,
            colors={1: CYAN, 2: BLUE, 3: MAGENTA, 4: YELLOW},
        ),
    )
    metric_line("query_k", format_counter_compact(Counter(row["query_k"] for row in stats), total))
    metric_line("rules", numeric_summary([row["total_rules"] for row in stats]))
    metric_line("category_pair", format_counter_compact(Counter(row["categories"] for row in stats), total))
    metric_line(
        "top_user",
        format_counter_compact(Counter(row["top_priority_user"] for row in stats), total),
    )

    label_mismatch = sum(not row["label_matches_top"] for row in stats)
    target_mismatch = sum(not row["target_matches_top"] for row in stats)
    check_color = GREEN if label_mismatch == 0 and target_mismatch == 0 else RED
    metric_line(
        "checks",
        color(
            f"label!=top {label_mismatch}, target!=top {target_mismatch}",
            check_color,
        ),
    )
    print()

combined = [row for stats in all_groups.values() for row in stats]
combined_total = len(combined)
if len(all_groups) > 1:
    title("Overall")
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
PY
