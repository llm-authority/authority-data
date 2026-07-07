"""Generate base synthetic authority datasets.

This script produces structured base rows only. Natural-language paraphrases of
``AuthoritySetting`` are expected to be created by ``make_paraphrase_data.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "base"
GENERAL_AUTHORITY = "GeneralAuthority"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attribute_sampling import attribute_sampling
from src.category_sampling import category_sampling, make_category_combinations
from src.domains import DEFAULT_CATEGORIES
from src.label_based_polarity_sampling import (
    deduplicate_rows,
    make_rows,
    polarity_sampling,
)


def _sample_expanded_authority_data(
    *,
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int = 2,
    num_random_fills: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)

    category_combinations = make_category_combinations(DEFAULT_CATEGORIES)
    sampled_categories = category_sampling(
        category_combinations,
        n=num_categories,
        rng=rng,
    )
    attribute_samples = attribute_sampling(
        sampled_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        rng=rng,
    )
    expanded_samples = polarity_sampling(
        attribute_samples,
        num_users=num_users,
        num_random_fills=num_random_fills,
        rng=rng,
    )

    counts = {
        "category_candidates": len(category_combinations),
        "sampled_categories": len(sampled_categories),
        "attribute_samples": len(attribute_samples),
        "polarity_expanded_samples": len(expanded_samples),
    }
    return expanded_samples, counts


def generate_authority_data(
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int = 2,
    num_random_fills: int = 2,
) -> tuple[list[dict], dict[str, int]]:
    """Return the legacy row shape used for quick previews."""

    expanded_samples, counts = _sample_expanded_authority_data(
        seed=seed,
        num_categories=num_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        num_users=num_users,
        num_random_fills=num_random_fills,
    )
    rows = make_rows(expanded_samples)
    deduplicated_rows = deduplicate_rows(rows)

    counts = counts | {
        "deduplicated_rows": len(deduplicated_rows),
        "dropped_duplicates": len(rows) - len(deduplicated_rows),
    }

    return deduplicated_rows, counts


def label_to_text(label: str) -> str:
    if label == "yes":
        return "Yes"
    if label == "no":
        return "No"
    raise ValueError(f"Unknown label: {label!r}")


def _rules_with_categories(
    sample: dict[str, Any],
    setting: dict[str, Any],
) -> list[dict[str, str]]:
    front_count = len(sample["original_front"])
    rules = []

    for index, (value, label) in enumerate(setting["rules"]):
        category = (
            sample["front_category"]
            if index < front_count
            else sample["back_category"]
        )
        rules.append(
            {
                "category": category,
                "value": value,
                "label": label,
            }
        )

    return rules


def make_base_authority_setting(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    tool_name: str | None = None,
) -> dict[str, Any]:
    authority_setting: dict[str, Any] = {
        "dataset": dataset_name,
        "users": [
            {
                "user": setting["user"],
                "authority": setting["authority"],
                "rules": _rules_with_categories(sample, setting),
            }
            for setting in sample["authority_setting"]
        ],
    }
    if tool_name is not None:
        authority_setting["tool"] = tool_name
    return authority_setting


def make_attribute_combination(
    sample: dict[str, Any],
    *,
    tool_name: str | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "attributes": {
            sample["front_category"]: sample["user1_selected"]["front"],
            sample["back_category"]: sample["user1_selected"]["back"],
        }
    }
    if tool_name is not None:
        query["tool"] = tool_name
    return query


def make_base_row(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    tool_name: str | None = None,
) -> dict[str, Any]:
    authority_setting = make_base_authority_setting(
        sample,
        dataset_name=dataset_name,
        tool_name=tool_name,
    )
    query = make_attribute_combination(sample, tool_name=tool_name)
    metadata = {
        "categories": list(sample["categories"]),
        "priority": list(sample["priority"]),
        "is_conflict": sample["user_conflict"] == "yes",
        "BaseAuthoritySetting": authority_setting,
        "ParaphraseVersion": None,
        "source": {
            "category_idx": sample["category_idx"],
            "k_pair_idx": sample["k_pair_idx"],
            "sample_idx": sample["sample_idx"],
            "case_id": sample["case_id"],
            "random_fill_idx": sample["random_fill_idx"],
        },
    }
    if tool_name is not None:
        metadata["tool"] = tool_name

    return {
        "AuthoritySetting": authority_setting,
        "Query": query,
        "Label": label_to_text(sample["label"]),
        "metadata": metadata,
    }


def _semantic_row_key(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    return json.dumps(
        {
            "AuthoritySetting": row["AuthoritySetting"],
            "Query": row["Query"],
            "Label": row["Label"],
            "categories": metadata["categories"],
            "priority": metadata["priority"],
            "is_conflict": metadata["is_conflict"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def deduplicate_base_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated = []
    seen = set()

    for row in rows:
        key = _semantic_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)

    return deduplicated


def add_row_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row_idx,
            **{key: value for key, value in row.items() if key != "id"},
        }
        for row_idx, row in enumerate(rows)
    ]


def generate_base_authority_datasets(
    *,
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int = 2,
    num_random_fills: int = 2,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    expanded_samples, sampling_counts = _sample_expanded_authority_data(
        seed=seed,
        num_categories=num_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        num_users=num_users,
        num_random_fills=num_random_fills,
    )

    datasets: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {"sampling": sampling_counts, "datasets": {}}

    rows = [
        make_base_row(sample, dataset_name=GENERAL_AUTHORITY)
        for sample in expanded_samples
    ]
    deduplicated = deduplicate_base_rows(rows)
    datasets[GENERAL_AUTHORITY] = add_row_ids(deduplicated)
    counts["datasets"][GENERAL_AUTHORITY] = {
        "rows_before_deduplication": len(rows),
        "rows": len(deduplicated),
        "dropped_duplicates": len(rows) - len(deduplicated),
    }

    return datasets, counts


def split_rows(
    rows: list[dict[str, Any]],
    *,
    test_ratio: float = 0.2,
    rng: random.Random | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = rng or random
    shuffled = list(rows)
    rng.shuffle(shuffled)

    if not shuffled:
        return {"train": [], "test": []}
    if test_ratio == 0:
        return {"train": shuffled, "test": []}
    if test_ratio == 1:
        return {"train": [], "test": shuffled}

    test_count = round(len(shuffled) * test_ratio)
    test_count = max(1, min(test_count, len(shuffled) - 1))
    return {
        "train": shuffled[test_count:],
        "test": shuffled[:test_count],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_splits(
    datasets: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    written_counts = {}

    for dataset_idx, (dataset_name, rows) in enumerate(datasets.items()):
        splits = split_rows(
            rows,
            test_ratio=test_ratio,
            rng=random.Random(seed + dataset_idx),
        )
        dataset_counts = {}
        for split_name, split_rows_ in splits.items():
            path = output_dir / dataset_name / f"{split_name}.jsonl"
            reindexed_rows = add_row_ids(split_rows_)
            write_jsonl(path, reindexed_rows)
            dataset_counts[split_name] = len(reindexed_rows)
        written_counts[dataset_name] = dataset_counts

    return written_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate base synthetic authority JSONL files."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-categories",
        type=int,
        default=None,
        help="Number of category pairs to sample. Defaults to all pairs.",
    )
    parser.add_argument("--max-k-pairs-per-category", type=int, default=10)
    parser.add_argument("--num-samples-per-k-pair", type=int, default=10)
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument("--num-random-fills", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print counts without writing JSONL files.",
    )
    parser.add_argument("--num-examples", type=int, default=1)
    return parser.parse_args()


def _rules_to_text(rules: list[list[str] | tuple[str, str]]) -> str:
    return "; ".join(f"{value}={label}" for value, label in rules)


def _format_cell(value: object, max_width: int) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > max_width:
        text = text[: max_width - 3] + "..."
    return text.ljust(max_width)


def _compact_json(value: object, max_width: int) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > max_width:
        text = text[: max_width - 3] + "..."
    return text


def print_examples_table(rows: list[dict], num_examples: int = 5) -> None:
    preview_rows = rows[:num_examples]
    if not preview_rows:
        print("Examples: no rows")
        return

    columns = [
        ("idx", 3),
        ("categories", 28),
        ("query", 24),
        ("priority", 16),
        ("label", 5),
        ("is_conflict", 11),
        ("authority_setting", 104),
    ]

    table_rows = []
    for idx, row in enumerate(preview_rows):
        table_row = dict(row)
        table_row["idx"] = idx
        table_row["categories"] = ", ".join(row["categories"])
        table_row["priority"] = ", ".join(row["priority"])
        table_row["authority_setting"] = " | ".join(
            f"{setting['user']}={setting['authority']}: "
            f"{_rules_to_text(setting['rules'])}"
            for setting in row["authority_setting"]
        )
        table_rows.append(table_row)

    header = " | ".join(_format_cell(name, width) for name, width in columns)
    separator = "-+-".join("-" * width for _, width in columns)
    print(f"Examples (first {len(preview_rows)})")
    print(header)
    print(separator)

    for row in table_rows:
        print(
            " | ".join(
                _format_cell(row[column_name], width)
                for column_name, width in columns
            )
        )


def print_base_examples_table(
    dataset_name: str,
    rows: list[dict[str, Any]],
    num_examples: int = 5,
) -> None:
    preview_rows = rows[:num_examples]
    if not preview_rows:
        print(f"{dataset_name} examples: no rows")
        return

    columns = [
        ("id", 34),
        ("label", 5),
        ("conflict", 8),
        ("query", 60),
        ("authority_setting", 96),
    ]

    table_rows = []
    for row in preview_rows:
        table_rows.append(
            {
                "id": row["id"],
                "label": row["Label"],
                "conflict": row["metadata"]["is_conflict"],
                "query": _compact_json(row["Query"], 60),
                "authority_setting": _compact_json(row["AuthoritySetting"], 96),
            }
        )

    header = " | ".join(_format_cell(name, width) for name, width in columns)
    separator = "-+-".join("-" * width for _, width in columns)
    print(f"{dataset_name} examples (first {len(preview_rows)})")
    print(header)
    print(separator)

    for row in table_rows:
        print(
            " | ".join(
                _format_cell(row[column_name], width)
                for column_name, width in columns
            )
        )


def main() -> None:
    args = parse_args()
    datasets, counts = generate_base_authority_datasets(
        seed=args.seed,
        num_categories=args.num_categories,
        max_k_pairs_per_category=args.max_k_pairs_per_category,
        num_samples_per_k_pair=args.num_samples_per_k_pair,
        num_users=args.num_users,
        num_random_fills=args.num_random_fills,
    )

    if not args.dry_run:
        counts["written"] = write_dataset_splits(
            datasets,
            output_dir=args.output_dir,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print("Counts")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print()
    for dataset_name, rows in datasets.items():
        print_base_examples_table(
            dataset_name,
            rows,
            num_examples=args.num_examples,
        )
        print()


if __name__ == "__main__":
    main()
