"""Attribute sampling from sampled category pairs."""

from __future__ import annotations

import random
from itertools import combinations, product
from typing import Any


def attribute_sampling(
    categories: list[dict[str, Any]],
    max_k_pairs_per_category: int = 10,
    num_samples_per_k_pair: int = 10,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Sample a random number of attributes for each sampled category pair."""

    rng = rng or random
    results: list[dict[str, Any]] = []

    for category_idx, category in enumerate(categories):
        front_values = list(category["front_candidates"])
        back_values = list(category["back_candidates"])

        possible_k_pairs = list(
            product(
                range(1, len(front_values) + 1),
                range(1, len(back_values) + 1),
            )
        )
        sampled_k_pairs = rng.sample(
            possible_k_pairs,
            k=min(max_k_pairs_per_category, len(possible_k_pairs)),
        )

        for k_pair_idx, (front_k, back_k) in enumerate(sampled_k_pairs):
            possible_front_samples = list(combinations(front_values, front_k))
            possible_back_samples = list(combinations(back_values, back_k))
            possible_attribute_samples = list(
                product(possible_front_samples, possible_back_samples)
            )

            sample_size = min(
                num_samples_per_k_pair,
                len(possible_attribute_samples),
            )
            sampled_attributes = rng.sample(
                possible_attribute_samples,
                k=sample_size,
            )

            for sample_idx, (sampled_front, sampled_back) in enumerate(
                sampled_attributes
            ):
                results.append(
                    {
                        "category_idx": category_idx,
                        "k_pair_idx": k_pair_idx,
                        "sample_idx": sample_idx,
                        "category_axes": category["category_axes"],
                        "categories": category["categories"],
                        "front_category": category["front_category"],
                        "back_category": category["back_category"],
                        "front_k": front_k,
                        "back_k": back_k,
                        "front_candidates": front_values,
                        "back_candidates": back_values,
                        "front": list(sampled_front),
                        "back": list(sampled_back),
                    }
                )

    return results
