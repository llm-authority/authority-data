"""Polarity sampling and final row construction."""

from __future__ import annotations

import json
import random
from typing import Any, Iterable


POLARITIES = ("yes", "no")
USER_IDS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + tuple(
    f"A{letter}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWX"
)


def parse_user_counts(num_users: int | str | Iterable[int]) -> list[int]:
    if isinstance(num_users, int):
        counts = [num_users]
    elif isinstance(num_users, str):
        counts = [
            int(part.strip())
            for part in num_users.split(",")
            if part.strip()
        ]
    else:
        counts = list(num_users)

    if not counts:
        raise ValueError("At least one user count is required.")

    normalized = []
    for count in counts:
        if not 1 <= count <= len(USER_IDS):
            raise ValueError(f"user counts must be between 1 and {len(USER_IDS)}")
        if count not in normalized:
            normalized.append(count)
    return normalized


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
    num_users: int | str | Iterable[int] = (1, 2, 3, 4),
    num_random_fills: int = 2,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Expand attribute samples into priority-aware yes/no label cases.

    For each user count, a random priority order is sampled. The highest-priority
    user's selected query label is the gold label. For two or more users,
    conflict and non-conflict cases are generated evenly: conflict means at
    least one lower-priority user assigns the opposite selected query label.
    """

    rng = rng or random
    user_counts = parse_user_counts(num_users)

    results: list[dict[str, Any]] = []

    for row_idx, sample in enumerate(samples):
        front_values = list(sample["front"])
        back_values = list(sample["back"])
        selected_front = rng.choice(front_values)
        selected_back = rng.choice(back_values)
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
                        "user_count": user_count,
                        "target_user": target_user,
                        "target_label": target_label,
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
                        "priority": priority,
                        "label": target_label,
                        "user_conflict": "yes" if has_conflict else "no",
                    }
                )

    return results


def make_case_specs(
    user_counts: list[int],
    *,
    rng: random.Random,
) -> list[tuple[int, str, bool]]:
    specs: list[tuple[int, str, bool]] = []
    conflict_capable_counts = [count for count in user_counts if count >= 2]

    for user_count in user_counts:
        conflict_options = (False,) if user_count == 1 else (False, True)
        for target_label in POLARITIES:
            for has_conflict in conflict_options:
                specs.append((user_count, target_label, has_conflict))

    non_conflict_count = sum(not has_conflict for _, _, has_conflict in specs)
    conflict_count = sum(has_conflict for _, _, has_conflict in specs)
    deficit = non_conflict_count - conflict_count
    if deficit > 0 and conflict_capable_counts:
        for index in range(deficit):
            specs.append(
                (
                    rng.choice(conflict_capable_counts),
                    POLARITIES[index % len(POLARITIES)],
                    True,
                )
            )

    rng.shuffle(specs)
    return specs


def opposite_label(label: str) -> str:
    if label == "yes":
        return "no"
    if label == "no":
        return "yes"
    raise ValueError(f"Unknown label: {label!r}")


def make_selected_labels(
    *,
    user_ids: list[str],
    priority: list[str],
    target_label: str,
    has_conflict: bool,
    rng: random.Random,
) -> dict[str, str]:
    target_user = priority[0]
    lower_priority_users = priority[1:]
    labels = {target_user: target_label}

    if not has_conflict:
        for user_id in lower_priority_users:
            labels[user_id] = target_label
        return labels

    if not lower_priority_users:
        raise ValueError("Conflict cases require at least two users.")

    conflict_user = rng.choice(lower_priority_users)
    conflict_label = opposite_label(target_label)
    for user_id in lower_priority_users:
        labels[user_id] = conflict_label if user_id == conflict_user else rng.choice(POLARITIES)

    return labels


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
