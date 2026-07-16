"""Build structured base authority data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.make_base_authority_data import (
    DEFAULT_OUTPUT_DIR as DEFAULT_BASE_OUTPUT_DIR,
    DEFAULT_V3_CONFLICT_RATIOS,
    DEFAULT_V3_MAX_RULES_PER_SCENARIO,
    DEFAULT_V3_TEST_NUM_USERS,
    DEFAULT_V3_TRAIN_NUM_USERS,
    GENERAL_AUTHORITY,
    TOOL_AUTHORITY,
    generate_base_authority_datasets,
    generate_v3_base_authority_dataset_splits,
    resolve_max_rules_per_scenario,
    resolve_split_user_counts,
    write_v3_dataset_splits,
    write_dataset_splits,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured base authority JSONL files."
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
    parser.add_argument(
        "--no-v3",
        action="store_true",
        help="Skip GeneralAuthorityV3 and ToolAuthorityV3 generation.",
    )
    parser.add_argument(
        "--v3-train-num-users",
        default=DEFAULT_V3_TRAIN_NUM_USERS,
        help="V3 train user counts. Supports comma lists or ranges. Default: 1-5.",
    )
    parser.add_argument(
        "--v3-test-num-users",
        default=DEFAULT_V3_TEST_NUM_USERS,
        help="V3 test user counts. Supports comma lists or ranges. Default: 5-50.",
    )
    parser.add_argument(
        "--v3-conflict-ratios",
        default=",".join(str(ratio) for ratio in DEFAULT_V3_CONFLICT_RATIOS),
        help="Comma-separated V3 lower-priority conflict ratios. Default: 0.1..0.9.",
    )
    parser.add_argument(
        "--v3-max-rules-per-scenario",
        type=int,
        default=DEFAULT_V3_MAX_RULES_PER_SCENARIO,
        help="Maximum total prompt rules for V3 rows. Default: 1000.",
    )
    parser.add_argument(
        "--general-v3-train-rows",
        type=int,
        default=500,
        help="Train rows to sample for GeneralAuthorityV3.",
    )
    parser.add_argument(
        "--general-v3-test-rows",
        type=int,
        default=1500,
        help="Test rows to sample for GeneralAuthorityV3.",
    )
    parser.add_argument(
        "--tool-v3-train-rows",
        type=int,
        default=1000,
        help="Train rows to sample for ToolAuthorityV3.",
    )
    parser.add_argument(
        "--tool-v3-test-rows",
        type=int,
        default=3000,
        help="Test rows to sample for ToolAuthorityV3.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--base-output-dir", type=Path, default=DEFAULT_BASE_OUTPUT_DIR)
    return parser.parse_args()


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

    print("Build base authority data")
    datasets, base_counts = generate_base_authority_datasets(
        seed=args.seed,
        num_categories=args.num_categories,
        max_k_pairs_per_category=args.max_k_pairs_per_category,
        num_samples_per_k_pair=args.num_samples_per_k_pair,
        num_users=generation_user_counts,
        num_random_fills=args.num_random_fills,
        max_rules_per_scenario=max_rules_per_scenario,
    )
    base_counts["written"] = write_dataset_splits(
        datasets,
        output_dir=args.base_output_dir,
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
    if not args.no_v3:
        v3_datasets, v3_counts = generate_v3_base_authority_dataset_splits(
            seed=args.seed,
            num_categories=args.num_categories,
            max_k_pairs_per_category=args.max_k_pairs_per_category,
            num_samples_per_k_pair=args.num_samples_per_k_pair,
            train_num_users=args.v3_train_num_users,
            test_num_users=args.v3_test_num_users,
            conflict_ratios=args.v3_conflict_ratios,
            num_random_fills=args.num_random_fills,
            max_rules_per_scenario=args.v3_max_rules_per_scenario,
            general_train_rows=args.general_v3_train_rows,
            general_test_rows=args.general_v3_test_rows,
            tool_train_rows=args.tool_v3_train_rows,
            tool_test_rows=args.tool_v3_test_rows,
        )
        v3_counts["written"] = write_v3_dataset_splits(
            v3_datasets,
            output_dir=args.base_output_dir,
        )
        base_counts["v3"] = v3_counts
    print(json.dumps(base_counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
