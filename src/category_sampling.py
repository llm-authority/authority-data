"""Category-pair sampling for flat category definitions."""

from __future__ import annotations

import random
from itertools import combinations, product
from typing import Any

from src.domains import DEFAULT_CATEGORIES, CategoryValues


def make_category_combinations(
    categories: CategoryValues = DEFAULT_CATEGORIES,
) -> list[dict[str, Any]]:
    """Build category pairs across different mutually exclusive axes."""

    category_pairs: list[dict[str, Any]] = []

    for (front_axis, front_options), (back_axis, back_options) in combinations(
        categories.items(),
        2,
    ):
        for front_category, back_category in product(
            front_options.keys(),
            back_options.keys(),
        ):
            category_pairs.append(
                {
                    "category_axes": [front_axis, back_axis],
                    "categories": [front_category, back_category],
                    "front_category": front_category,
                    "back_category": back_category,
                    "front_candidates": list(front_options[front_category]),
                    "back_candidates": list(back_options[back_category]),
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
