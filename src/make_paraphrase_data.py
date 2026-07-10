"""Create synthetic paraphrases from base authority JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "base"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "paraphrase"
DEFAULT_PARAPHRASE_VERSION = "synthetic_v1"
DEFAULT_PROMPT_VERSION = "synthetic_prompt_v6"
DATASET_NAMES = ("GeneralAuthorityV1", "ToolAuthorityV1")
SPLIT_NAMES = ("train", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _text_value(value: str) -> str:
    if value.isupper():
        return value
    return value.replace("_", " ")


CATEGORY_DISPLAY_NAMES = {
    "day": "Day",
    "date": "Date",
    "time": "Time",
    "month": "Month",
    "year": "Year",
    "recipient": "Recipient",
    "information_type": "Information type",
    "purpose": "Purpose",
}


def _category_display_name(category: str) -> str:
    return CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " ").capitalize())


def _display_value(category: str, value: str) -> str:
    text = _text_value(value)
    if category in {"day", "month"}:
        return text.title()
    return text


def _user_display_name(user: str) -> str:
    return f"User {user}"


def _group_rules_by_category(rules: list[dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"yes": [], "no": []}
    )
    for rule in rules:
        grouped[rule["category"]][rule["label"]].append(rule["value"])
    return grouped


def _join_values(category: str, values: list[str]) -> str:
    return ", ".join(_display_value(category, value) for value in values)


def _format_rule_items(
    grouped_rules: dict[str, dict[str, list[str]]],
    label: str,
) -> list[str]:
    items = []
    for category, labels in grouped_rules.items():
        values = labels[label]
        if values:
            items.append(
                f"- {_category_display_name(category)}: {_join_values(category, values)}"
            )
    if not items:
        items.append("- none")
    return items


def render_user_policy(user_setting: dict[str, Any]) -> str:
    grouped = _group_rules_by_category(user_setting["rules"])
    user = user_setting["user"]
    lines = [
        _user_display_name(user),
        "Allowed:",
        *_format_rule_items(grouped, "yes"),
        "Disallowed:",
        *_format_rule_items(grouped, "no"),
    ]
    return "\n".join(lines)


def render_query(query: dict[str, Any], *, include_tool: bool = True) -> str:
    attributes = query.get("attributes", {})
    lines = []
    tool = query.get("tool")
    if include_tool and tool:
        lines.append(f"- tool: {tool}")
    for category, value in attributes.items():
        lines.append(
            f"- {_category_display_name(category)}: {_display_value(category, value)}"
        )
    return "\n".join(lines)


def render_synthetic_paraphrase(base_row: dict[str, Any]) -> str:
    authority_setting = base_row["AuthoritySetting"]
    priority = base_row["metadata"]["priority"]
    tool = authority_setting.get("tool")
    is_tool_authority = str(authority_setting.get("dataset", "")).startswith(
        "ToolAuthority"
    )

    if is_tool_authority:
        scope = f"Task: decide whether the requested use of the {tool} tool is authorized."
    elif tool:
        scope = f"Task: decide whether {tool} is authorized."
    else:
        scope = "Task: decide whether the requested action is authorized."
    instruction = (
        "Multiple users provide allow/disallow rules for the requested action. "
        "If their rules disagree, follow the priority order."
    )
    priority_text = " > ".join(_user_display_name(user) for user in priority)
    user_text = "\n\n".join(
        render_user_policy(user_setting)
        for user_setting in authority_setting["users"]
    )
    query_text = render_query(base_row["Query"], include_tool=not is_tool_authority)
    query_header = "Query conditions" if is_tool_authority else "Query"
    return (
        f"{scope}\n"
        f"{instruction}\n"
        f"Priority: {priority_text}\n\n"
        f"{user_text}\n\n"
        f"{query_header}:\n{query_text}"
    )


def make_paraphrase_row(
    base_row: dict[str, Any],
    *,
    row_id: int,
    paraphrase_version: str = DEFAULT_PARAPHRASE_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    return {
        "text": render_synthetic_paraphrase(base_row),
        "label": base_row["Label"],
        "meta_data": {
            "query": base_row["Query"],
            "paraphrase_version": paraphrase_version,
        },
    }


def make_paraphrase_rows(
    base_rows: list[dict[str, Any]],
    *,
    paraphrase_version: str = DEFAULT_PARAPHRASE_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    return [
        make_paraphrase_row(
            base_row,
            row_id=row_idx,
            paraphrase_version=paraphrase_version,
            prompt_version=prompt_version,
        )
        for row_idx, base_row in enumerate(base_rows)
    ]


def make_paraphrase_data(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_names: tuple[str, ...] = DATASET_NAMES,
    split_names: tuple[str, ...] = SPLIT_NAMES,
    paraphrase_version: str = DEFAULT_PARAPHRASE_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}

    for dataset_name in dataset_names:
        counts[dataset_name] = {}
        for split_name in split_names:
            input_path = input_dir / dataset_name / f"{split_name}.jsonl"
            if not input_path.exists():
                continue

            base_rows = read_jsonl(input_path)
            paraphrase_rows = make_paraphrase_rows(
                base_rows,
                paraphrase_version=paraphrase_version,
                prompt_version=prompt_version,
            )
            output_path = output_dir / dataset_name / f"{split_name}.jsonl"
            write_jsonl(output_path, paraphrase_rows)
            counts[dataset_name][split_name] = len(paraphrase_rows)

    return counts


def _format_cell(value: object, max_width: int) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > max_width:
        text = text[: max_width - 3] + "..."
    return text.ljust(max_width)


def _print_field_value_row(
    example_id: object,
    field: str,
    value: object,
    *,
    example_width: int = 7,
    field_width: int = 22,
) -> None:
    lines = str(value).splitlines() or [""]
    for line_idx, line in enumerate(lines):
        example_cell = str(example_id) if line_idx == 0 else ""
        field_cell = field if line_idx == 0 else ""
        print(
            f"{example_cell.ljust(example_width)} | "
            f"{field_cell.ljust(field_width)} | "
            f"{line}"
        )


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("meta_data") or row.get("metadata") or {}


def _row_label(row: dict[str, Any]) -> str:
    return row.get("label") or row.get("Label") or ""


def _row_id(row: dict[str, Any]) -> object:
    metadata = _row_metadata(row)
    return metadata.get("RowId", row.get("id", ""))


def _row_attribute_combination(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _row_metadata(row)
    return (
        row.get("AttributeCombination")
        or row.get("Query")
        or metadata.get("query")
        or metadata.get("AttributeBase")
        or {}
    )


def _make_summary_table_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _row_metadata(row)
    return {
        "id": _row_id(row),
        "label": _row_label(row),
        "conflict": metadata.get("is_conflict", ""),
        "attribute_combination": json.dumps(
            _row_attribute_combination(row),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "text": row["text"],
    }


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("id", 4),
        ("label", 5),
        ("conflict", 8),
        ("attribute_combination", 52),
        ("text", 96),
    ]
    header = " | ".join(_format_cell(name, width) for name, width in columns)
    separator = "-+-".join("-" * width for _, width in columns)
    print(header)
    print(separator)
    for row in rows:
        table_row = _make_summary_table_row(row)
        print(
            " | ".join(
                _format_cell(table_row[column_name], width)
                for column_name, width in columns
            )
        )


def _print_full_table(rows: list[dict[str, Any]]) -> None:
    print("example | field                  | value")
    print("--------+------------------------+----------------------------------------")
    for row in rows:
        metadata = _row_metadata(row)
        example_id = _row_id(row)
        _print_field_value_row(example_id, "id", example_id)
        _print_field_value_row(example_id, "label", _row_label(row))
        _print_field_value_row(
            example_id,
            "is_conflict",
            metadata.get("is_conflict", ""),
        )
        _print_field_value_row(
            example_id,
            "AttributeCombination",
            json.dumps(
                _row_attribute_combination(row),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        _print_field_value_row(
            example_id,
            "metadata",
            json.dumps(
                {
                    "query": _row_attribute_combination(row),
                    "paraphrase_version": metadata.get(
                        "paraphrase_version",
                        metadata.get("ParaphraseVersion"),
                    ),
                    "query_meaning_score": metadata.get("query_meaning_score"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        _print_field_value_row(
            example_id,
            "text",
            row["text"],
        )
        print("--------+------------------------+----------------------------------------")


def print_paraphrase_examples_table(
    dataset_name: str,
    rows: list[dict[str, Any]],
    *,
    num_table_examples: int = 5,
    num_full_examples: int = 1,
) -> None:
    table_rows = rows[:num_table_examples]
    full_rows = rows[:num_full_examples]
    if not table_rows and not full_rows:
        print(f"{dataset_name} examples: no rows")
        return

    if table_rows:
        print(f"{dataset_name} summary examples (first {len(table_rows)})")
        _print_summary_table(table_rows)
        print()

    if full_rows:
        print(f"{dataset_name} full example (first {len(full_rows)})")
        _print_full_table(full_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synthetic paraphrases from base authority JSONL files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paraphrase-version", default=DEFAULT_PARAPHRASE_VERSION)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--num-table-examples", type=int, default=5)
    parser.add_argument("--num-full-examples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = make_paraphrase_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        paraphrase_version=args.paraphrase_version,
        prompt_version=args.prompt_version,
    )

    print("Counts")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print()

    for dataset_name in DATASET_NAMES:
        path = args.output_dir / dataset_name / "train.jsonl"
        if not path.exists():
            continue
        print_paraphrase_examples_table(
            dataset_name,
            read_jsonl(path),
            num_table_examples=args.num_table_examples,
            num_full_examples=args.num_full_examples,
        )
        print()


if __name__ == "__main__":
    main()
