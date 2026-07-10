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
DEFAULT_PROMPT_VERSION = "synthetic_prompt_v1"
DATASET_NAMES = ("GeneralAuthority", "ToolAuthority")
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


def _title_value(value: str) -> str:
    if value.isupper():
        return value
    return value.replace("_", " ")


def _group_rules_by_category(rules: list[dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"yes": [], "no": []}
    )
    for rule in rules:
        grouped[rule["category"]][rule["label"]].append(rule["value"])
    return grouped


def _join_values(values: list[str]) -> str:
    return ", ".join(_title_value(value) for value in values)


def _format_rule_items(
    user: str,
    grouped_rules: dict[str, dict[str, list[str]]],
    label: str,
) -> list[str]:
    items = []
    for category, labels in grouped_rules.items():
        values = labels[label]
        if values:
            items.append(
                f"- {user} {category.replace('_', ' ')}: {_join_values(values)}"
            )
    if not items:
        items.append(f"- {user}: none")
    return items


def render_user_policy(user_setting: dict[str, Any]) -> str:
    grouped = _group_rules_by_category(user_setting["rules"])
    user = user_setting["user"]
    lines = [
        user,
        "Allowed:",
        *_format_rule_items(user, grouped, "yes"),
        "Disallowed:",
        *_format_rule_items(user, grouped, "no"),
    ]
    return "\n".join(lines)


def render_query(query: dict[str, Any]) -> str:
    attributes = query.get("attributes", {})
    lines = []
    tool = query.get("tool")
    if tool:
        lines.append(f"- tool: {tool}")
    for category, value in attributes.items():
        lines.append(f"- {category.replace('_', ' ')}: {_title_value(value)}")
    return "\n".join(lines)


def render_synthetic_paraphrase(base_row: dict[str, Any]) -> str:
    authority_setting = base_row["AuthoritySetting"]
    priority = base_row["metadata"]["priority"]
    tool = authority_setting.get("tool")

    scope = f"Task: decide whether {tool} is authorized." if tool else (
        "Task: decide whether the requested action is authorized."
    )
    priority_text = " > ".join(priority)
    user_text = "\n\n".join(
        render_user_policy(user_setting)
        for user_setting in authority_setting["users"]
    )
    query_text = render_query(base_row["Query"])
    return (
        f"{scope}\n"
        f"Priority: {priority_text}\n\n"
        f"{user_text}\n\n"
        f"Query:\n{query_text}"
    )


def make_paraphrase_row(
    base_row: dict[str, Any],
    *,
    row_id: int,
    paraphrase_version: str = DEFAULT_PARAPHRASE_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    metadata = dict(base_row["metadata"])
    metadata["is_conflict"] = bool(base_row["metadata"]["is_conflict"])
    metadata["BaseAuthoritySetting"] = base_row["AuthoritySetting"]
    metadata["AttributeBase"] = base_row["Query"]
    metadata["ParaphraseVersion"] = paraphrase_version
    metadata["PromptVersion"] = prompt_version
    metadata["BaseRowId"] = base_row["id"]

    return {
        "id": row_id,
        "text": render_synthetic_paraphrase(base_row),
        "AttributeCombination": base_row["Query"],
        "Label": base_row["Label"],
        "metadata": metadata,
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


def _make_summary_table_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "label": row["Label"],
        "conflict": row["metadata"]["is_conflict"],
        "attribute_combination": json.dumps(
            row["AttributeCombination"],
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
        example_id = row["id"]
        _print_field_value_row(example_id, "id", row["id"])
        _print_field_value_row(example_id, "label", row["Label"])
        _print_field_value_row(
            example_id,
            "is_conflict",
            row["metadata"]["is_conflict"],
        )
        _print_field_value_row(
            example_id,
            "AttributeCombination",
            json.dumps(
                row["AttributeCombination"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        _print_field_value_row(
            example_id,
            "metadata",
            json.dumps(
                {
                    "is_conflict": row["metadata"]["is_conflict"],
                    "PromptVersion": row["metadata"]["PromptVersion"],
                    "ParaphraseVersion": row["metadata"]["ParaphraseVersion"],
                    "AttributeBase": row["metadata"]["AttributeBase"],
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
