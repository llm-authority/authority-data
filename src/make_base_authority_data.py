"""Generate base synthetic authority datasets.

This script produces structured base rows only. Natural-language paraphrases of
``AuthoritySetting`` are expected to be created by ``make_paraphrase_data.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "base"
GENERAL_AUTHORITY = "GeneralAuthorityV1"
TOOL_AUTHORITY = "ToolAuthorityV1"
GENERAL_AUTHORITY_V3 = "GeneralAuthorityV3"
TOOL_AUTHORITY_V3 = "ToolAuthorityV3"
DEFAULT_TRAIN_NUM_USERS = "1,2,3"
DEFAULT_TEST_NUM_USERS = "1,2,3,4,5"
DEFAULT_V3_TRAIN_NUM_USERS = "1-5"
DEFAULT_V3_TEST_NUM_USERS = "5-50"
DEFAULT_V3_CONFLICT_RATIOS = tuple(round(index / 10, 1) for index in range(1, 10))
DEFAULT_V3_MAX_RULES_PER_SCENARIO = 1000

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.category_sampling import category_sampling, make_category_combinations
from src.domains import (
    ALL_TOOL_NAMES,
    DEFAULT_CATEGORIES,
    GENERAL_AUTHORITY_TEST_CATEGORIES,
    GENERAL_AUTHORITY_TRAIN_CATEGORIES,
    TEST_TOOL_NAMES,
    TOOL_AUTHORITY_CATEGORIES,
    TOOL_AUTHORITY_TEST_CATEGORIES,
    TOOL_AUTHORITY_TRAIN_CATEGORIES,
    TRAIN_TOOL_NAMES,
    CategoryValues,
)
from src.label_based_polarity_sampling import (
    USER_IDS,
    label_items,
    make_case_specs,
    make_selected_labels,
    opposite_label,
    parse_user_counts,
)


def progress_iter(iterable, *, desc: str, total: int | None = None, unit: str = "it"):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        disable=not sys.stderr.isatty(),
    )


def _sample_expanded_authority_data(
    *,
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int | str | Iterable[int] = (1, 2, 3, 4),
    num_random_fills: int = 2,
    max_rules_per_scenario: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_rules_per_scenario is not None and max_rules_per_scenario < 1:
        raise ValueError("max_rules_per_scenario must be at least 1")

    rng = random.Random(seed)

    category_combinations = make_multi_category_combinations(
        DEFAULT_CATEGORIES,
        category_counts=range(2, len(DEFAULT_CATEGORIES) + 1),
    )
    sampled_categories = category_sampling(
        category_combinations,
        n=num_categories,
        rng=rng,
    )
    attribute_samples = multi_attribute_sampling(
        sampled_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        rng=rng,
    )
    expanded_samples = polarity_sampling_multi(
        attribute_samples,
        num_users=num_users,
        num_random_fills=num_random_fills,
        rng=rng,
    )
    rule_limited_samples = [
        sample
        for sample in expanded_samples
        if max_rules_per_scenario is None
        or count_scenario_rules(sample["authority_setting"])
        <= max_rules_per_scenario
    ]

    counts = {
        "category_candidates": len(category_combinations),
        "sampled_categories": len(sampled_categories),
        "attribute_samples": len(attribute_samples),
        "polarity_expanded_samples": len(expanded_samples),
        "rule_limited_samples": len(rule_limited_samples),
        "dropped_by_rule_limit": len(expanded_samples) - len(rule_limited_samples),
    }
    return rule_limited_samples, counts


def make_multi_category_combinations(
    categories: CategoryValues,
    category_counts: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Build category combinations across non-empty categories."""

    non_empty_categories = [
        (category, values)
        for category, values in categories.items()
        if values
    ]
    if not non_empty_categories:
        return []

    if category_counts is None:
        category_counts = (len(non_empty_categories),)

    category_combinations = []
    for category_count in category_counts:
        if category_count < 1:
            continue
        for category_group in combinations(non_empty_categories, category_count):
            category_combinations.append(
                {
                    "category_axes": [category for category, _ in category_group],
                    "categories": [category for category, _ in category_group],
                    "candidates_by_category": {
                        category: list(values)
                        for category, values in category_group
                    },
                }
            )

    return category_combinations


def multi_attribute_sampling(
    categories: list[dict[str, Any]],
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Sample attribute subsets for category combinations with 3+ categories."""

    rng = rng or random
    results: list[dict[str, Any]] = []

    for category_idx, category in progress_iter(
        enumerate(categories),
        desc="sample attributes",
        total=len(categories),
        unit="category",
    ):
        category_names = list(category["categories"])
        candidates_by_category = {
            name: list(category["candidates_by_category"][name])
            for name in category_names
        }
        possible_k_tuples = list(
            product(
                *(
                    range(1, len(candidates_by_category[name]) + 1)
                    for name in category_names
                )
            )
        )
        sampled_k_tuples = rng.sample(
            possible_k_tuples,
            k=min(max_k_pairs_per_category, len(possible_k_tuples)),
        )

        for k_pair_idx, k_tuple in enumerate(sampled_k_tuples):
            possible_samples_by_category = [
                list(combinations(candidates_by_category[name], k))
                for name, k in zip(category_names, k_tuple)
            ]
            sampled_attributes = sample_attribute_products(
                possible_samples_by_category,
                sample_size=num_samples_per_k_pair,
                rng=rng,
            )

            for sample_idx, sampled_values in enumerate(sampled_attributes):
                category_values = {
                    name: list(values)
                    for name, values in zip(category_names, sampled_values)
                }
                results.append(
                    {
                        "category_idx": category_idx,
                        "k_pair_idx": k_pair_idx,
                        "sample_idx": sample_idx,
                        "category_axes": category["category_axes"],
                        "categories": category_names,
                        "category_ks": dict(zip(category_names, k_tuple)),
                        "category_candidates": candidates_by_category,
                        "category_values": category_values,
                    }
                )

    return results


def sample_attribute_products(
    possible_samples_by_category: list[list[tuple[str, ...]]],
    *,
    sample_size: int,
    rng: random.Random,
) -> list[tuple[tuple[str, ...], ...]]:
    """Sample from a product of per-category combinations without materializing it."""

    if sample_size < 1 or not possible_samples_by_category:
        return []

    total_size = 1
    for samples in possible_samples_by_category:
        total_size *= len(samples)
    target_size = min(sample_size, total_size)

    sampled = []
    seen = set()
    max_attempts = max(100, target_size * 20)
    attempts = 0

    while len(sampled) < target_size and attempts < max_attempts:
        attempts += 1
        candidate = tuple(
            rng.choice(samples)
            for samples in possible_samples_by_category
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        sampled.append(candidate)

    if len(sampled) == target_size:
        return sampled

    for candidate in product(*possible_samples_by_category):
        if candidate in seen:
            continue
        sampled.append(candidate)
        if len(sampled) == target_size:
            break

    return sampled


def polarity_sampling_multi(
    samples: list[dict[str, Any]],
    num_users: int | str | Iterable[int] = (1, 2, 3, 4),
    num_random_fills: int = 2,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Expand multi-attribute samples into priority-aware yes/no cases."""

    rng = rng or random
    user_counts = parse_user_counts(num_users)
    results: list[dict[str, Any]] = []

    for row_idx, sample in progress_iter(
        enumerate(samples),
        desc="expand polarity",
        total=len(samples),
        unit="sample",
    ):
        category_values = {
            category: list(values)
            for category, values in sample["category_values"].items()
        }
        selected_attributes = {
            category: rng.choice(values)
            for category, values in category_values.items()
        }
        case_specs = make_case_specs(user_counts, rng=rng)

        for case_id, (user_count, target_label, has_conflict) in enumerate(case_specs):
            user_ids = list(USER_IDS[:user_count])
            for random_fill_idx in range(num_random_fills):
                priority = rng.sample(user_ids, k=len(user_ids))
                target_user = priority[0]
                selected_labels = make_selected_labels(
                    user_ids=user_ids,
                    priority=priority,
                    target_label=target_label,
                    has_conflict=has_conflict,
                    rng=rng,
                )
                authority_setting = []

                for user_id in user_ids:
                    selected_label = selected_labels[user_id]
                    rules = []
                    for category, values in category_values.items():
                        rules.extend(
                            {
                                "category": category,
                                "value": value,
                                "label": label,
                            }
                            for value, label in label_items(
                                values,
                                selected_attributes[category],
                                selected_label,
                                rng=rng,
                            )
                        )
                    authority_setting.append(
                        {
                            "user": user_id,
                            "authority": selected_label,
                            "rules": rules,
                        }
                    )

                results.append(
                    {
                        "original_row_idx": row_idx,
                        "category_idx": sample["category_idx"],
                        "k_pair_idx": sample["k_pair_idx"],
                        "sample_idx": sample["sample_idx"],
                        "case_id": case_id,
                        "random_fill_idx": random_fill_idx,
                        "user_count": user_count,
                        "target_user": target_user,
                        "target_label": target_label,
                        "category_axes": sample["category_axes"],
                        "categories": sample["categories"],
                        "category_candidates": sample["category_candidates"],
                        "category_ks": sample["category_ks"],
                        "category_values": category_values,
                        "user1_selected": {"attributes": selected_attributes},
                        "authority_setting": authority_setting,
                        "priority": priority,
                        "label": target_label,
                        "user_conflict": "yes" if has_conflict else "no",
                    }
                )

    return results


def _sample_expanded_tool_authority_data(
    *,
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int | str | Iterable[int] = (1, 2, 3, 4),
    num_random_fills: int = 2,
    max_rules_per_scenario: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_rules_per_scenario is not None and max_rules_per_scenario < 1:
        raise ValueError("max_rules_per_scenario must be at least 1")

    rng = random.Random(seed)
    category_combinations = make_multi_category_combinations(
        TOOL_AUTHORITY_CATEGORIES,
        category_counts=range(2, len(TOOL_AUTHORITY_CATEGORIES) + 1),
    )
    sampled_categories = category_sampling(
        category_combinations,
        n=num_categories,
        rng=rng,
    )
    attribute_samples = multi_attribute_sampling(
        sampled_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        rng=rng,
    )
    expanded_samples = polarity_sampling_multi(
        attribute_samples,
        num_users=num_users,
        num_random_fills=num_random_fills,
        rng=rng,
    )
    for sample in expanded_samples:
        sample["tool_name"] = rng.choice(ALL_TOOL_NAMES)

    rule_limited_samples = [
        sample
        for sample in expanded_samples
        if max_rules_per_scenario is None
        or count_scenario_rules(sample["authority_setting"])
        <= max_rules_per_scenario
    ]

    counts = {
        "category_candidates": len(category_combinations),
        "sampled_categories": len(sampled_categories),
        "attribute_samples": len(attribute_samples),
        "polarity_expanded_samples": len(expanded_samples),
        "rule_limited_samples": len(rule_limited_samples),
        "dropped_by_rule_limit": len(expanded_samples) - len(rule_limited_samples),
    }
    return rule_limited_samples, counts


def count_scenario_rules(authority_setting: list[dict[str, Any]]) -> int:
    """Return the total number of rules across every user in a scenario."""

    return sum(len(setting["rules"]) for setting in authority_setting)


def resolve_split_user_counts(
    *,
    num_users: int | str | Iterable[int] | None = None,
    train_num_users: int | str | Iterable[int] | None = None,
    test_num_users: int | str | Iterable[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve split-specific user counts and their generation union."""

    train_counts = parse_user_counts(
        train_num_users
        if train_num_users is not None
        else (num_users if num_users is not None else DEFAULT_TRAIN_NUM_USERS)
    )
    test_counts = parse_user_counts(
        test_num_users
        if test_num_users is not None
        else (num_users if num_users is not None else DEFAULT_TEST_NUM_USERS)
    )
    generation_counts = list(dict.fromkeys(train_counts + test_counts))
    return train_counts, test_counts, generation_counts


def max_rules_per_user() -> int:
    """Return the largest number of attribute rules one user can receive."""

    return max(
        len(category_pair["front_candidates"])
        + len(category_pair["back_candidates"])
        for category_pair in make_category_combinations(DEFAULT_CATEGORIES)
    )


def resolve_max_rules_per_scenario(
    max_rules_per_scenario: int | None,
    *,
    test_user_counts: Iterable[int],
) -> int:
    """Use the requested limit or the maximum supported by the test schema."""

    if max_rules_per_scenario is not None:
        if max_rules_per_scenario < 1:
            raise ValueError("max_rules_per_scenario must be at least 1")
        return max_rules_per_scenario

    user_counts = list(test_user_counts)
    if not user_counts:
        raise ValueError("At least one test user count is required.")
    return max_rules_per_user() * max(user_counts)


def generate_authority_data(
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int | str | Iterable[int] = (1, 2, 3, 4),
    num_random_fills: int = 2,
    max_rules_per_scenario: int | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Return the legacy row shape used for quick previews."""

    expanded_samples, counts = _sample_expanded_authority_data(
        seed=seed,
        num_categories=num_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        num_users=num_users,
        num_random_fills=num_random_fills,
        max_rules_per_scenario=max_rules_per_scenario,
    )
    rows = [
        {
            "categories": sample["categories"],
            "authority_setting": sample["authority_setting"],
            "query": ", ".join(
                str(value)
                for value in sample["user1_selected"]["attributes"].values()
            ),
            "priority": sample["priority"],
            "label": sample["label"],
            "is_conflict": sample["user_conflict"],
        }
        for sample in expanded_samples
    ]
    deduplicated_rows = deduplicate_legacy_rows(rows)

    counts = counts | {
        "deduplicated_rows": len(deduplicated_rows),
        "dropped_duplicates": len(rows) - len(deduplicated_rows),
    }

    return deduplicated_rows, counts


def deduplicate_legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated = []
    seen = set()

    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)

    return deduplicated


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
    if setting["rules"] and isinstance(setting["rules"][0], dict):
        return [dict(rule) for rule in setting["rules"]]

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
    if "attributes" in sample["user1_selected"]:
        query = {"attributes": dict(sample["user1_selected"]["attributes"])}
        if tool_name is not None:
            query["tool"] = tool_name
        return query

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
        "RuleCount": count_scenario_rules(sample["authority_setting"]),
        "source": {
            "category_idx": sample["category_idx"],
            "k_pair_idx": sample["k_pair_idx"],
            "sample_idx": sample["sample_idx"],
            "case_id": sample["case_id"],
            "random_fill_idx": sample["random_fill_idx"],
            "user_count": sample["user_count"],
            "target_user": sample["target_user"],
            "target_label": sample["target_label"],
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

    for row in progress_iter(
        rows,
        desc="deduplicate rows",
        total=len(rows),
        unit="row",
    ):
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


def summarize_base_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "labels": dict(Counter(row["Label"] for row in rows)),
        "conflict": dict(Counter(str(row["metadata"]["is_conflict"]) for row in rows)),
        "user_count": dict(
            Counter(str(len(row["AuthoritySetting"]["users"])) for row in rows)
        ),
    }


def parse_user_count_spec(value: int | str | Iterable[int]) -> list[int]:
    if isinstance(value, str) and "-" in value:
        start_text, sep, end_text = value.partition("-")
        if sep and start_text.strip() and end_text.strip():
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"Invalid user count range: {value!r}")
            return parse_user_counts(range(start, end + 1))
    return parse_user_counts(value)


def parse_conflict_ratios(value: str | Iterable[float]) -> list[float]:
    ratios = (
        [
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
        if isinstance(value, str)
        else list(value)
    )
    if not ratios:
        raise ValueError("At least one conflict ratio is required.")
    normalized = []
    for ratio in ratios:
        if not 0 < ratio < 1:
            raise ValueError("Conflict ratios must be between 0 and 1.")
        rounded = round(ratio, 3)
        if rounded not in normalized:
            normalized.append(rounded)
    return normalized


def category_subset(
    categories: CategoryValues,
    selected_categories: Iterable[str],
) -> CategoryValues:
    return {
        category: list(categories[category])
        for category in selected_categories
    }


def make_v3_rule_template(
    category_values: dict[str, list[str]],
    selected_attributes: dict[str, str],
    selected_label: str,
    filler_labels: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    rules = []
    for category, values in category_values.items():
        selected_value = selected_attributes[category]
        for value in values:
            label = (
                selected_label
                if value == selected_value
                else filler_labels[(category, value)]
            )
            rules.append(
                {
                    "category": category,
                    "value": value,
                    "label": label,
                }
            )
    return rules


def v3_conflict_count(user_count: int, conflict_ratio: float) -> int:
    lower_priority_count = user_count - 1
    if lower_priority_count < 1:
        raise ValueError("V3 conflict rows require at least two users.")
    return max(1, min(lower_priority_count, round(lower_priority_count * conflict_ratio)))


def make_v3_row(
    *,
    dataset_name: str,
    sample: dict[str, Any],
    user_count: int,
    target_label: str,
    conflict_ratio: float,
    random_fill_idx: int,
    case_id: int,
    rng: random.Random,
    tool_name_list: list[str],
    max_rules_per_scenario: int | None,
) -> dict[str, Any] | None:
    category_values = {
        category: list(values)
        for category, values in sample["category_values"].items()
    }
    per_user_rule_count = sum(len(values) for values in category_values.values())
    rule_count = per_user_rule_count * user_count
    if max_rules_per_scenario is not None and rule_count > max_rules_per_scenario:
        return None

    selected_attributes = {
        category: rng.choice(values)
        for category, values in category_values.items()
    }
    filler_labels = {
        (category, value): rng.choice(("yes", "no"))
        for category, values in category_values.items()
        for value in values
        if value != selected_attributes[category]
    }
    opposite = opposite_label(target_label)
    rule_templates = {
        target_label: make_v3_rule_template(
            category_values,
            selected_attributes,
            target_label,
            filler_labels,
        ),
        opposite: make_v3_rule_template(
            category_values,
            selected_attributes,
            opposite,
            filler_labels,
        ),
    }
    user_ids = list(USER_IDS[:user_count])
    priority = rng.sample(user_ids, k=len(user_ids))
    target_user = priority[0]
    lower_priority_users = priority[1:]
    conflict_user_count = v3_conflict_count(user_count, conflict_ratio)
    conflict_users = set(rng.sample(lower_priority_users, k=conflict_user_count))
    authority_setting = []
    for user_id in user_ids:
        selected_label = opposite if user_id in conflict_users else target_label
        authority_setting.append(
            {
                "user": user_id,
                "authority": selected_label,
                "rules": [
                    dict(rule)
                    for rule in rule_templates[selected_label]
                ],
            }
        )

    tool_name = rng.choice(tool_name_list) if tool_name_list else None
    row = make_base_row(
        {
            "category_idx": sample["category_idx"],
            "k_pair_idx": sample["k_pair_idx"],
            "sample_idx": sample["sample_idx"],
            "case_id": case_id,
            "random_fill_idx": random_fill_idx,
            "user_count": user_count,
            "target_user": target_user,
            "target_label": target_label,
            "category_axes": sample["category_axes"],
            "categories": sample["categories"],
            "category_candidates": sample["category_candidates"],
            "category_ks": sample["category_ks"],
            "category_values": category_values,
            "user1_selected": {"attributes": selected_attributes},
            "authority_setting": authority_setting,
            "priority": priority,
            "label": target_label,
            "user_conflict": "yes",
        },
        dataset_name=dataset_name,
        tool_name=tool_name,
    )
    actual_conflict_ratio = conflict_user_count / (user_count - 1)
    row["metadata"]["v3"] = {
        "main_user": target_user,
        "user_count": user_count,
        "lower_priority_user_count": user_count - 1,
        "requested_conflict_ratio": conflict_ratio,
        "actual_conflict_ratio": actual_conflict_ratio,
        "conflict_user_count": conflict_user_count,
        "agreeing_lower_priority_user_count": (
            user_count - 1 - conflict_user_count
        ),
    }
    return row


def v3_candidate_rule_count(sample: dict[str, Any], user_count: int) -> int:
    return user_count * sum(
        len(values)
        for values in sample["category_values"].values()
    )


def generate_v3_sampled_split_rows(
    *,
    dataset_name: str,
    categories: CategoryValues,
    user_counts: Iterable[int],
    conflict_ratios: Iterable[float],
    seed: int,
    num_categories: int | None,
    max_k_pairs_per_category: int,
    num_samples_per_k_pair: int,
    num_random_fills: int,
    max_rules_per_scenario: int | None,
    target_count: int,
    tool_names: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count < 0:
        raise ValueError("target split row counts must be non-negative")
    if max_rules_per_scenario is not None and max_rules_per_scenario < 1:
        raise ValueError("max_rules_per_scenario must be at least 1")

    rng = random.Random(seed)
    usable_user_counts = [count for count in user_counts if count >= 2]
    skipped_user_counts = [count for count in user_counts if count < 2]
    ratios = list(conflict_ratios)
    category_combinations = make_multi_category_combinations(
        categories,
        category_counts=range(2, len(categories) + 1),
    )
    sampled_categories = category_sampling(
        category_combinations,
        n=num_categories,
        rng=rng,
    )
    attribute_samples = multi_attribute_sampling(
        sampled_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        rng=rng,
    )
    tool_name_list = list(tool_names) if tool_names is not None else []

    feasible_pairs = [
        (sample, user_count)
        for sample in attribute_samples
        for user_count in usable_user_counts
        if max_rules_per_scenario is None
        or v3_candidate_rule_count(sample, user_count) <= max_rules_per_scenario
    ]
    total_candidate_count = (
        len(attribute_samples)
        * len(usable_user_counts)
        * 2
        * len(ratios)
        * num_random_fills
    )
    feasible_candidate_count = (
        len(feasible_pairs)
        * 2
        * len(ratios)
        * num_random_fills
    )
    skipped_by_rule_limit = total_candidate_count - feasible_candidate_count
    if target_count and not feasible_pairs:
        raise ValueError(
            f"No feasible V3 rows for {dataset_name}; try increasing "
            "--v3-max-rules-per-scenario or reducing user/category settings."
        )

    ratio_plan: list[float] = []
    while len(ratio_plan) < target_count:
        ratio_batch = list(ratios)
        rng.shuffle(ratio_batch)
        ratio_plan.extend(ratio_batch)
    ratio_plan = ratio_plan[:target_count]
    rng.shuffle(ratio_plan)

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    duplicate_attempts = 0
    max_attempts = max(100, target_count * 50)
    attempts = 0
    progress_bar = progress_iter(
        ratio_plan,
        desc=f"sample {dataset_name} rows",
        total=len(ratio_plan),
        unit="row",
    )
    progress = iter(progress_bar)
    while len(rows) < target_count and attempts < max_attempts:
        conflict_ratio = (
            ratio_plan[len(rows)]
            if len(rows) < len(ratio_plan)
            else rng.choice(ratios)
        )
        attempts += 1
        sample, user_count = rng.choice(feasible_pairs)
        row = make_v3_row(
            dataset_name=dataset_name,
            sample=sample,
            user_count=user_count,
            target_label=rng.choice(("yes", "no")),
            conflict_ratio=conflict_ratio,
            random_fill_idx=attempts % max(1, num_random_fills),
            case_id=attempts - 1,
            rng=rng,
            tool_name_list=tool_name_list,
            max_rules_per_scenario=max_rules_per_scenario,
        )
        if row is None:
            continue
        key = _semantic_row_key(row)
        if key in seen_keys:
            duplicate_attempts += 1
            continue
        seen_keys.add(key)
        rows.append(row)
        try:
            next(progress)
        except StopIteration:
            pass

    if hasattr(progress_bar, "close"):
        progress_bar.close()
    if len(rows) < target_count:
        raise ValueError(
            f"Only generated {len(rows)} unique V3 rows for {dataset_name}, "
            f"but {target_count} were requested."
        )

    counts = {
        "category_candidates": len(category_combinations),
        "sampled_categories": len(sampled_categories),
        "attribute_samples": len(attribute_samples),
        "candidate_rows_estimate": total_candidate_count,
        "feasible_candidate_rows_estimate": feasible_candidate_count,
        "rows_before_deduplication": attempts - duplicate_attempts,
        "rows": len(rows),
        "dropped_duplicates": duplicate_attempts,
        "skipped_user_counts": skipped_user_counts,
        "skipped_by_rule_limit": skipped_by_rule_limit,
        "max_rules_per_scenario": max_rules_per_scenario,
        "distribution": summarize_base_rows(rows),
    }
    return rows, counts


def generate_v3_split_rows(
    *,
    dataset_name: str,
    categories: CategoryValues,
    user_counts: Iterable[int],
    conflict_ratios: Iterable[float],
    seed: int,
    num_categories: int | None,
    max_k_pairs_per_category: int,
    num_samples_per_k_pair: int,
    num_random_fills: int,
    max_rules_per_scenario: int | None,
    tool_names: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_rules_per_scenario is not None and max_rules_per_scenario < 1:
        raise ValueError("max_rules_per_scenario must be at least 1")

    rng = random.Random(seed)
    usable_user_counts = [count for count in user_counts if count >= 2]
    skipped_user_counts = [count for count in user_counts if count < 2]
    ratios = list(conflict_ratios)
    category_combinations = make_multi_category_combinations(
        categories,
        category_counts=range(2, len(categories) + 1),
    )
    sampled_categories = category_sampling(
        category_combinations,
        n=num_categories,
        rng=rng,
    )
    attribute_samples = multi_attribute_sampling(
        sampled_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        rng=rng,
    )
    tool_name_list = list(tool_names) if tool_names is not None else []

    rows = []
    case_id = 0
    skipped_by_rule_limit = 0
    desc = f"generate {dataset_name} rows"
    for sample in progress_iter(
        attribute_samples,
        desc=desc,
        total=len(attribute_samples),
        unit="sample",
    ):
        category_values = {
            category: list(values)
            for category, values in sample["category_values"].items()
        }
        selected_attributes = {
            category: rng.choice(values)
            for category, values in category_values.items()
        }
        filler_labels = {
            (category, value): rng.choice(("yes", "no"))
            for category, values in category_values.items()
            for value in values
            if value != selected_attributes[category]
        }

        for user_count in usable_user_counts:
            user_ids = list(USER_IDS[:user_count])
            for target_label in ("yes", "no"):
                opposite = opposite_label(target_label)
                rule_templates = {
                    target_label: make_v3_rule_template(
                        category_values,
                        selected_attributes,
                        target_label,
                        filler_labels,
                    ),
                    opposite: make_v3_rule_template(
                        category_values,
                        selected_attributes,
                        opposite,
                        filler_labels,
                    ),
                }
                for conflict_ratio in ratios:
                    for random_fill_idx in range(num_random_fills):
                        priority = rng.sample(user_ids, k=len(user_ids))
                        target_user = priority[0]
                        lower_priority_users = priority[1:]
                        conflict_user_count = v3_conflict_count(
                            user_count,
                            conflict_ratio,
                        )
                        conflict_users = set(
                            rng.sample(lower_priority_users, k=conflict_user_count)
                        )
                        authority_setting = []
                        for user_id in user_ids:
                            selected_label = (
                                opposite if user_id in conflict_users else target_label
                            )
                            authority_setting.append(
                                {
                                    "user": user_id,
                                    "authority": selected_label,
                                    "rules": [
                                        dict(rule)
                                        for rule in rule_templates[selected_label]
                                    ],
                                }
                            )

                        rule_count = count_scenario_rules(authority_setting)
                        if (
                            max_rules_per_scenario is not None
                            and rule_count > max_rules_per_scenario
                        ):
                            skipped_by_rule_limit += 1
                            continue

                        tool_name = rng.choice(tool_name_list) if tool_name_list else None
                        row = make_base_row(
                            {
                                "category_idx": sample["category_idx"],
                                "k_pair_idx": sample["k_pair_idx"],
                                "sample_idx": sample["sample_idx"],
                                "case_id": case_id,
                                "random_fill_idx": random_fill_idx,
                                "user_count": user_count,
                                "target_user": target_user,
                                "target_label": target_label,
                                "category_axes": sample["category_axes"],
                                "categories": sample["categories"],
                                "category_candidates": sample["category_candidates"],
                                "category_ks": sample["category_ks"],
                                "category_values": category_values,
                                "user1_selected": {"attributes": selected_attributes},
                                "authority_setting": authority_setting,
                                "priority": priority,
                                "label": target_label,
                                "user_conflict": "yes",
                            },
                            dataset_name=dataset_name,
                            tool_name=tool_name,
                        )
                        actual_conflict_ratio = conflict_user_count / (
                            user_count - 1
                        )
                        row["metadata"]["v3"] = {
                            "main_user": target_user,
                            "user_count": user_count,
                            "lower_priority_user_count": user_count - 1,
                            "requested_conflict_ratio": conflict_ratio,
                            "actual_conflict_ratio": actual_conflict_ratio,
                            "conflict_user_count": conflict_user_count,
                            "agreeing_lower_priority_user_count": (
                                user_count - 1 - conflict_user_count
                            ),
                        }
                        rows.append(row)
                        case_id += 1

    deduplicated = deduplicate_base_rows(rows)
    counts = {
        "category_candidates": len(category_combinations),
        "sampled_categories": len(sampled_categories),
        "attribute_samples": len(attribute_samples),
        "rows_before_deduplication": len(rows),
        "rows": len(deduplicated),
        "dropped_duplicates": len(rows) - len(deduplicated),
        "skipped_user_counts": skipped_user_counts,
        "skipped_by_rule_limit": skipped_by_rule_limit,
        "max_rules_per_scenario": max_rules_per_scenario,
        "distribution": summarize_base_rows(deduplicated),
    }
    return deduplicated, counts


def generate_v3_base_authority_dataset_splits(
    *,
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    train_num_users: int | str | Iterable[int] = DEFAULT_V3_TRAIN_NUM_USERS,
    test_num_users: int | str | Iterable[int] = DEFAULT_V3_TEST_NUM_USERS,
    conflict_ratios: str | Iterable[float] = DEFAULT_V3_CONFLICT_RATIOS,
    num_random_fills: int = 2,
    max_rules_per_scenario: int | None = DEFAULT_V3_MAX_RULES_PER_SCENARIO,
    general_train_rows: int | None = 500,
    general_test_rows: int | None = 1500,
    tool_train_rows: int | None = 1000,
    tool_test_rows: int | None = 3000,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    train_user_counts = parse_user_count_spec(train_num_users)
    test_user_counts = parse_user_count_spec(test_num_users)
    ratios = parse_conflict_ratios(conflict_ratios)
    split_specs = {
        GENERAL_AUTHORITY_V3: {
            "train": {
                "categories": category_subset(
                    DEFAULT_CATEGORIES,
                    GENERAL_AUTHORITY_TRAIN_CATEGORIES,
                ),
                "user_counts": train_user_counts,
                "target_count": general_train_rows,
                "tool_names": None,
                "seed": seed + 3000,
            },
            "test": {
                "categories": category_subset(
                    DEFAULT_CATEGORIES,
                    GENERAL_AUTHORITY_TEST_CATEGORIES,
                ),
                "user_counts": test_user_counts,
                "target_count": general_test_rows,
                "tool_names": None,
                "seed": seed + 3100,
            },
        },
        TOOL_AUTHORITY_V3: {
            "train": {
                "categories": category_subset(
                    TOOL_AUTHORITY_CATEGORIES,
                    TOOL_AUTHORITY_TRAIN_CATEGORIES,
                ),
                "user_counts": train_user_counts,
                "target_count": tool_train_rows,
                "tool_names": TRAIN_TOOL_NAMES,
                "seed": seed + 4000,
            },
            "test": {
                "categories": category_subset(
                    TOOL_AUTHORITY_CATEGORIES,
                    TOOL_AUTHORITY_TEST_CATEGORIES,
                ),
                "user_counts": test_user_counts,
                "target_count": tool_test_rows,
                "tool_names": TEST_TOOL_NAMES,
                "seed": seed + 4100,
            },
        },
    }

    datasets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    counts: dict[str, Any] = {
        "conflict_ratios": ratios,
        "train_user_counts": train_user_counts,
        "test_user_counts": test_user_counts,
        "datasets": {},
    }
    for dataset_name, split_spec in progress_iter(
        split_specs.items(),
        desc="generate V3 datasets",
        total=len(split_specs),
        unit="dataset",
    ):
        datasets[dataset_name] = {}
        counts["datasets"][dataset_name] = {}
        for split_name, spec in progress_iter(
            split_spec.items(),
            desc=f"generate {dataset_name} splits",
            total=len(split_spec),
            unit="split",
        ):
            if spec["target_count"] is None:
                rows, split_counts = generate_v3_split_rows(
                    dataset_name=dataset_name,
                    categories=spec["categories"],
                    user_counts=spec["user_counts"],
                    conflict_ratios=ratios,
                    seed=spec["seed"],
                    num_categories=num_categories,
                    max_k_pairs_per_category=max_k_pairs_per_category,
                    num_samples_per_k_pair=num_samples_per_k_pair,
                    num_random_fills=num_random_fills,
                    max_rules_per_scenario=max_rules_per_scenario,
                    tool_names=spec["tool_names"],
                )
                sampled_rows = rows
            else:
                sampled_rows, split_counts = generate_v3_sampled_split_rows(
                    dataset_name=dataset_name,
                    categories=spec["categories"],
                    user_counts=spec["user_counts"],
                    conflict_ratios=ratios,
                    seed=spec["seed"],
                    num_categories=num_categories,
                    max_k_pairs_per_category=max_k_pairs_per_category,
                    num_samples_per_k_pair=num_samples_per_k_pair,
                    num_random_fills=num_random_fills,
                    max_rules_per_scenario=max_rules_per_scenario,
                    target_count=spec["target_count"],
                    tool_names=spec["tool_names"],
                )
            reindexed_rows = add_row_ids(sampled_rows)
            datasets[dataset_name][split_name] = reindexed_rows
            counts["datasets"][dataset_name][split_name] = split_counts | {
                "written_rows": len(reindexed_rows),
                "target_count": spec["target_count"],
                "written_distribution": summarize_base_rows(reindexed_rows),
            }
    return datasets, counts


def write_v3_dataset_splits(
    datasets: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, int]]:
    written_counts = {}
    for dataset_name, splits in datasets.items():
        written_counts[dataset_name] = {}
        for split_name, rows in splits.items():
            path = output_dir / dataset_name / f"{split_name}.jsonl"
            write_jsonl(path, rows)
            written_counts[dataset_name][split_name] = len(rows)
    return written_counts


def sample_v3_split_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int | None,
    conflict_ratios: Iterable[float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count is None:
        return list(rows)
    if target_count < 0:
        raise ValueError("target split row counts must be non-negative")
    if target_count > len(rows):
        raise ValueError(
            f"Requested {target_count} rows, but only {len(rows)} rows are available."
        )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    rows_by_ratio: dict[float, list[int]] = {}
    for index, row in enumerate(rows):
        ratio = row["metadata"]["v3"]["requested_conflict_ratio"]
        rows_by_ratio.setdefault(ratio, []).append(index)

    for ratio in conflict_ratios:
        candidates = rows_by_ratio.get(ratio, [])
        if not candidates or len(selected_indices) >= target_count:
            continue
        index = rng.choice(candidates)
        selected_indices.append(index)
        selected_set.add(index)

    remaining_indices = [
        index
        for index in range(len(rows))
        if index not in selected_set
    ]
    rng.shuffle(remaining_indices)
    selected_indices.extend(
        remaining_indices[: target_count - len(selected_indices)]
    )
    rng.shuffle(selected_indices)
    return [rows[index] for index in selected_indices]


def generate_base_authority_datasets(
    *,
    seed: int = 42,
    num_categories: int | None = None,
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    num_users: int | str | Iterable[int] = (1, 2, 3, 4),
    num_random_fills: int = 2,
    max_rules_per_scenario: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    expanded_samples, sampling_counts = _sample_expanded_authority_data(
        seed=seed,
        num_categories=num_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        num_users=num_users,
        num_random_fills=num_random_fills,
        max_rules_per_scenario=max_rules_per_scenario,
    )
    expanded_tool_samples, tool_sampling_counts = _sample_expanded_tool_authority_data(
        seed=seed + 1000,
        num_categories=num_categories,
        max_k_pairs_per_category=max_k_pairs_per_category,
        num_samples_per_k_pair=num_samples_per_k_pair,
        num_users=num_users,
        num_random_fills=num_random_fills,
        max_rules_per_scenario=max_rules_per_scenario,
    )

    datasets: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {
        "sampling": {
            GENERAL_AUTHORITY: sampling_counts,
            TOOL_AUTHORITY: tool_sampling_counts,
        },
        "datasets": {},
    }

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
        "distribution": summarize_base_rows(deduplicated),
    }

    tool_rows = [
        make_base_row(
            sample,
            dataset_name=TOOL_AUTHORITY,
            tool_name=sample["tool_name"],
        )
        for sample in expanded_tool_samples
    ]
    tool_deduplicated = deduplicate_base_rows(tool_rows)
    datasets[TOOL_AUTHORITY] = add_row_ids(tool_deduplicated)
    counts["datasets"][TOOL_AUTHORITY] = {
        "rows_before_deduplication": len(tool_rows),
        "rows": len(tool_deduplicated),
        "dropped_duplicates": len(tool_rows) - len(tool_deduplicated),
        "distribution": summarize_base_rows(tool_deduplicated),
    }

    return datasets, counts


def split_rows(
    rows: list[dict[str, Any]],
    *,
    test_ratio: float = 0.2,
    max_rules_per_scenario: int | None = None,
    train_user_counts: Iterable[int] | None = None,
    test_user_counts: Iterable[int] | None = None,
    train_categories: Iterable[str] | None = None,
    test_categories: Iterable[str] | None = None,
    train_tools: Iterable[str] | None = None,
    test_tools: Iterable[str] | None = None,
    rng: random.Random | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if max_rules_per_scenario is not None and max_rules_per_scenario < 1:
        raise ValueError("max_rules_per_scenario must be at least 1")

    rng = rng or random

    train_user_counts_set = (
        set(train_user_counts) if train_user_counts is not None else None
    )
    test_user_counts_set = (
        set(test_user_counts) if test_user_counts is not None else None
    )
    train_categories_set = set(train_categories) if train_categories is not None else None
    test_categories_set = set(test_categories) if test_categories is not None else None
    train_tools_set = set(train_tools) if train_tools is not None else None
    test_tools_set = set(test_tools) if test_tools is not None else None

    train_only_rows = []
    test_only_rows = []
    shared_rows = []
    train_rule_limit = (
        max_rules_per_scenario // 2
        if max_rules_per_scenario is not None
        else None
    )

    for row in rows:
        rule_count = row["metadata"]["RuleCount"]
        user_count = len(row["AuthoritySetting"]["users"])
        row_categories = set(row["metadata"]["categories"])
        row_tool = row["metadata"].get("tool")
        train_eligible = (
            (train_rule_limit is None or rule_count <= train_rule_limit)
            and (train_user_counts_set is None or user_count in train_user_counts_set)
            and (
                train_categories_set is None
                or row_categories.issubset(train_categories_set)
            )
            and (train_tools_set is None or row_tool in train_tools_set)
        )
        test_eligible = (
            (max_rules_per_scenario is None or rule_count <= max_rules_per_scenario)
            and (test_user_counts_set is None or user_count in test_user_counts_set)
            and (
                test_categories_set is None
                or row_categories.issubset(test_categories_set)
            )
            and (test_tools_set is None or row_tool in test_tools_set)
        )

        if train_eligible and test_eligible:
            shared_rows.append(row)
        elif train_eligible:
            train_only_rows.append(row)
        elif test_eligible:
            test_only_rows.append(row)

    rng.shuffle(shared_rows)
    rng.shuffle(train_only_rows)
    rng.shuffle(test_only_rows)

    shared_test_count = round(len(shared_rows) * test_ratio)
    if shared_rows and 0 < test_ratio < 1:
        shared_test_count = max(1, min(shared_test_count, len(shared_rows) - 1))

    return {
        "train": train_only_rows + shared_rows[shared_test_count:],
        "test": test_only_rows + shared_rows[:shared_test_count],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_split_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int | None = None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count is None:
        return list(rows)
    if target_count < 0:
        raise ValueError("target split row counts must be non-negative")
    if target_count > len(rows):
        raise ValueError(
            f"Requested {target_count} rows, but only {len(rows)} rows are available."
        )

    sampled_rows = list(rows)
    rng.shuffle(sampled_rows)
    return sampled_rows[:target_count]


def write_dataset_splits(
    datasets: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_ratio: float = 0.2,
    max_rules_per_scenario: int | None = None,
    train_user_counts: Iterable[int] | None = None,
    test_user_counts: Iterable[int] | None = None,
    train_rows_per_dataset: int | None = None,
    test_rows_per_dataset: int | None = None,
    train_rows_by_dataset: dict[str, int | None] | None = None,
    test_rows_by_dataset: dict[str, int | None] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    written_counts = {}

    for dataset_idx, (dataset_name, rows) in enumerate(datasets.items()):
        train_categories = None
        test_categories = None
        train_tools = None
        test_tools = None
        if dataset_name == GENERAL_AUTHORITY:
            train_categories = GENERAL_AUTHORITY_TRAIN_CATEGORIES
            test_categories = GENERAL_AUTHORITY_TEST_CATEGORIES
        elif dataset_name == TOOL_AUTHORITY:
            train_categories = TOOL_AUTHORITY_TRAIN_CATEGORIES
            test_categories = TOOL_AUTHORITY_TEST_CATEGORIES
            train_tools = TRAIN_TOOL_NAMES
            test_tools = TEST_TOOL_NAMES

        splits = split_rows(
            rows,
            test_ratio=test_ratio,
            max_rules_per_scenario=max_rules_per_scenario,
            train_user_counts=train_user_counts,
            test_user_counts=test_user_counts,
            train_categories=train_categories,
            test_categories=test_categories,
            train_tools=train_tools,
            test_tools=test_tools,
            rng=random.Random(seed + dataset_idx),
        )
        dataset_counts = {}
        for split_name, split_rows_ in splits.items():
            path = output_dir / dataset_name / f"{split_name}.jsonl"
            if split_name == "train":
                target_count = (
                    train_rows_by_dataset.get(dataset_name, train_rows_per_dataset)
                    if train_rows_by_dataset is not None
                    else train_rows_per_dataset
                )
            else:
                target_count = (
                    test_rows_by_dataset.get(dataset_name, test_rows_per_dataset)
                    if test_rows_by_dataset is not None
                    else test_rows_per_dataset
                )
            sampled_rows = sample_split_rows(
                split_rows_,
                target_count=target_count,
                rng=random.Random(seed + dataset_idx * 1000 + len(split_name)),
            )
            reindexed_rows = add_row_ids(sampled_rows)
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
        default=20,
        help="Number of category combinations to sample. Defaults to 20.",
    )
    parser.add_argument("--max-k-pairs-per-category", type=int, default=10)
    parser.add_argument("--num-samples-per-k-pair", type=int, default=10)
    parser.add_argument(
        "--num-users",
        default=None,
        help="Comma-separated user counts for both splits unless overridden.",
    )
    parser.add_argument(
        "--train-num-users",
        default=None,
        help="Comma-separated user counts allowed in train. Default: 1,2,3.",
    )
    parser.add_argument(
        "--test-num-users",
        default=None,
        help="Comma-separated user counts allowed in test. Default: 1,2,3,4,5.",
    )
    parser.add_argument("--num-random-fills", type=int, default=2)
    parser.add_argument(
        "--max-rules-per-scenario",
        type=int,
        default=None,
        help="Maximum total rules; defaults to the maximum supported by test users.",
    )
    parser.add_argument(
        "--train-rows-per-dataset",
        type=int,
        default=None,
        help="Fallback train rows to sample per dataset config after splitting.",
    )
    parser.add_argument(
        "--test-rows-per-dataset",
        type=int,
        default=None,
        help="Fallback test rows to sample per dataset config after splitting.",
    )
    parser.add_argument(
        "--general-train-rows",
        type=int,
        default=500,
        help="Train rows to sample for GeneralAuthorityV1.",
    )
    parser.add_argument(
        "--general-test-rows",
        type=int,
        default=1500,
        help="Test rows to sample for GeneralAuthorityV1.",
    )
    parser.add_argument(
        "--tool-train-rows",
        type=int,
        default=1000,
        help="Train rows to sample for ToolAuthorityV1.",
    )
    parser.add_argument(
        "--tool-test-rows",
        type=int,
        default=3000,
        help="Test rows to sample for ToolAuthorityV1.",
    )
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
    train_user_counts, test_user_counts, generation_user_counts = (
        resolve_split_user_counts(
            num_users=args.num_users,
            train_num_users=args.train_num_users,
            test_num_users=args.test_num_users,
        )
    )
    max_rules_per_scenario = resolve_max_rules_per_scenario(
        args.max_rules_per_scenario,
        test_user_counts=test_user_counts,
    )
    datasets, counts = generate_base_authority_datasets(
        seed=args.seed,
        num_categories=args.num_categories,
        max_k_pairs_per_category=args.max_k_pairs_per_category,
        num_samples_per_k_pair=args.num_samples_per_k_pair,
        num_users=generation_user_counts,
        num_random_fills=args.num_random_fills,
        max_rules_per_scenario=max_rules_per_scenario,
    )

    if not args.dry_run:
        counts["written"] = write_dataset_splits(
            datasets,
            output_dir=args.output_dir,
            test_ratio=args.test_ratio,
            max_rules_per_scenario=max_rules_per_scenario,
            train_user_counts=train_user_counts,
            test_user_counts=test_user_counts,
            train_rows_per_dataset=args.train_rows_per_dataset,
            test_rows_per_dataset=args.test_rows_per_dataset,
            train_rows_by_dataset={
                GENERAL_AUTHORITY: args.general_train_rows,
                TOOL_AUTHORITY: args.tool_train_rows,
            },
            test_rows_by_dataset={
                GENERAL_AUTHORITY: args.general_test_rows,
                TOOL_AUTHORITY: args.tool_test_rows,
            },
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
