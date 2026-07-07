"""Polarity sampling and final row construction."""

from __future__ import annotations

import json
import random
from itertools import product
from typing import Any


POLARITIES = ("yes", "no")
USER_IDS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def label_items(
    values: list[str],
    selected_value: str,
    selected_label: str,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """Set the query-matched value to a fixed label and randomize the rest."""

    rng = rng or random
    labeled = []

    for value in values:
        if value == selected_value:
            labeled.append((value, selected_label))
        else:
            labeled.append((value, rng.choice(POLARITIES)))

    return labeled


def polarity_sampling(
    samples: list[dict[str, Any]],
    num_users: int = 2,
    num_random_fills: int = 2,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Expand attribute samples into yes/no label cases.

    The selected query attributes get the fixed label for each user. All other
    attributes receive random yes/no labels.
    """

    rng = rng or random
    if not 1 <= num_users <= len(USER_IDS):
        raise ValueError(f"num_users must be between 1 and {len(USER_IDS)}")

    results: list[dict[str, Any]] = []
    user_ids = list(USER_IDS[:num_users])
    fixed_cases = list(product(POLARITIES, repeat=num_users))

    for row_idx, sample in enumerate(samples):
        front_values = list(sample["front"])
        back_values = list(sample["back"])
        selected_front = rng.choice(front_values)
        selected_back = rng.choice(back_values)

        for case_id, selected_labels in enumerate(fixed_cases):
            for random_fill_idx in range(num_random_fills):
                authority_setting = []

                for user_id, selected_label in zip(user_ids, selected_labels):
                    authority_setting.append(
                        {
                            "user": user_id,
                            "authority": selected_label,
                            "rules": label_items(
                                front_values,
                                selected_front,
                                selected_label,
                                rng=rng,
                            )
                            + label_items(
                                back_values,
                                selected_back,
                                selected_label,
                                rng=rng,
                            ),
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
                        "category_axes": sample["category_axes"],
                        "categories": sample["categories"],
                        "front_category": sample["front_category"],
                        "back_category": sample["back_category"],
                        "front_candidates": sample["front_candidates"],
                        "back_candidates": sample["back_candidates"],
                        "front_k": sample["front_k"],
                        "back_k": sample["back_k"],
                        "original_front": front_values,
                        "original_back": back_values,
                        "user1_selected": {
                            "front": selected_front,
                            "back": selected_back,
                        },
                        "authority_setting": authority_setting,
                        "priority": user_ids,
                        "label": selected_labels[0],
                        "user_conflict": "yes" if len(set(selected_labels)) > 1 else "no",
                    }
                )

    return results


def make_rows(
    expanded_samples: list[dict[str, Any]],
    authority_setting_as_json_string: bool = False,
    priority_as_json_string: bool = False,
) -> list[dict[str, Any]]:
    """Convert expanded samples into final dataset rows."""

    rows: list[dict[str, Any]] = []

    for sample in expanded_samples:
        authority_setting = sample["authority_setting"]
        priority = sample["priority"]
        query = (
            f"{sample['user1_selected']['front']}, "
            f"{sample['user1_selected']['back']}"
        )

        if authority_setting_as_json_string:
            authority_setting = json.dumps(authority_setting, ensure_ascii=False)
        if priority_as_json_string:
            priority = json.dumps(priority, ensure_ascii=False)

        rows.append(
            {
                "categories": sample["categories"],
                "authority_setting": authority_setting,
                "query": query,
                "priority": priority,
                "label": sample["label"],
                "is_conflict": sample["user_conflict"],
            }
        )

    return rows


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate rows by their full canonical JSON representation."""

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)

    return deduplicated
