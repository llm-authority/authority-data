"""Category-pair sampling for flat category definitions."""

from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from src.domains import DEFAULT_CATEGORIES, CategoryValues


def make_category_combinations(
    categories: CategoryValues = DEFAULT_CATEGORIES,
) -> list[dict[str, Any]]:
    """Build category pairs across non-empty categories."""

    category_pairs: list[dict[str, Any]] = []
    non_empty_categories = [
        (category, values)
        for category, values in categories.items()
        if values
    ]

    for (front_category, front_values), (back_category, back_values) in combinations(
        non_empty_categories,
        2,
    ):
        category_pairs.append(
            {
                "category_axes": [front_category, back_category],
                "categories": [front_category, back_category],
                "front_category": front_category,
                "back_category": back_category,
                "front_candidates": list(front_values),
                "back_candidates": list(back_values),
            }
        )

    return category_pairs


def category_sampling(
    category_combinations: list[dict[str, Any]] | None = None,
    n: int | None = 1,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Randomly sample category pairs.

    If ``n`` is None, every candidate is returned in shuffled order.
    """

    rng = rng or random
    candidates = list(
        category_combinations
        if category_combinations is not None
        else make_category_combinations()
    )

    if n is None:
        rng.shuffle(candidates)
        return candidates

    return rng.sample(candidates, k=min(n, len(candidates)))
