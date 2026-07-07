#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/paraphrase}"
DATASET="${DATASET:-}"
SPLIT="${SPLIT:-}"
SEED="${SEED:-}"

python - "$DATA_DIR" "$DATASET" "$SPLIT" "$SEED" <<'PY'
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any


data_dir = Path(sys.argv[1])
dataset_filter = sys.argv[2]
split_filter = sys.argv[3]
seed_arg = sys.argv[4]
use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def color(text: object, code: str) -> str:
    value = str(text)
    if not use_color:
        return value
    return f"\033[{code}m{value}\033[0m"


BOLD = "1"
DIM = "2"
GREEN = "32"
YELLOW = "33"
CYAN = "36"
RED = "31"
FIELD = "1;36"
SECTION = "1;35"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            row["_path"] = str(path)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def authority_setting(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("AuthoritySetting") or row.get("metadata", {}).get("BaseAuthoritySetting", {})


def query_attributes(row: dict[str, Any]) -> dict[str, Any]:
    query = row.get("AttributeCombination") or row.get("Query") or {}
    return query.get("attributes") or {}


def user_authority_map(row: dict[str, Any]) -> dict[str, str]:
    users = authority_setting(row).get("users") or []
    return {user.get("user", ""): user.get("authority", "") for user in users}


def section(title: str) -> None:
    print()
    print(color(title, SECTION))


def field(name: str, value: object) -> None:
    print(f"{color((name + ':').ljust(18), FIELD)} {value}")


paths = sorted(data_dir.glob("*/*.jsonl"))
if dataset_filter:
    paths = [path for path in paths if path.parent.name == dataset_filter]
if split_filter:
    paths = [path for path in paths if path.stem == split_filter]

if not paths:
    print(f"No matching JSONL files found under {data_dir}")
    raise SystemExit(0)

rows = []
for path in paths:
    rows.extend(read_jsonl(path))

if not rows:
    print(f"No rows found under {data_dir}")
    raise SystemExit(0)

rng = random.Random(int(seed_arg)) if seed_arg else random.SystemRandom()
row = rng.choice(rows)
metadata = row.get("metadata") or {}
priority = metadata.get("priority") or []
users = user_authority_map(row)
top_user = priority[0] if priority else ""
top_authority = users.get(top_user, "")
label = row.get("Label", "")
label_color = GREEN if label == "Yes" else YELLOW
conflict = bool(metadata.get("is_conflict", False))
conflict_color = RED if conflict else GREEN

print(color("Random Authority Sample", BOLD))
field("path", row["_path"])
field("line", row["_line_number"])
field("id", row.get("id", ""))
field("label", color(label, label_color))
field("conflict", color(conflict, conflict_color))
field("priority", " > ".join(priority))
field("top authority", f"{top_user}={top_authority}" if top_user else "")
field("users", ", ".join(f"{user}={authority}" for user, authority in users.items()))
field("query", json.dumps(query_attributes(row), ensure_ascii=False))
field("categories", ", ".join(metadata.get("categories") or []))

source = metadata.get("source") or {}
if source:
    field(
        "source",
        ", ".join(
            f"{key}={source[key]}"
            for key in (
                "user_count",
                "target_user",
                "target_label",
                "case_id",
                "random_fill_idx",
            )
            if key in source
        ),
    )

section("Prompt Text")
print(row.get("text") or json.dumps(row.get("AuthoritySetting"), ensure_ascii=False, indent=2))

section("Gold Answer")
print(color(f"<answer>{label}</answer>", CYAN))
PY
