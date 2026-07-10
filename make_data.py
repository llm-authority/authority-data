"""Run the base and synthetic paraphrase data builders."""

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
    generate_base_authority_datasets,
    resolve_max_rules_per_scenario,
    resolve_split_user_counts,
    write_dataset_splits,
)
from src.make_paraphrase_data import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PARAPHRASE_OUTPUT_DIR,
    DEFAULT_PROMPT_VERSION,
    DATASET_NAMES,
    make_paraphrase_data,
    print_paraphrase_examples_table,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build base authority data, synthetic paraphrases, and print previews."
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
<<<<<<< HEAD
        default="1,2,3",
        help="Comma-separated user counts allowed in train. Defaults to --num-users.",
    )
    parser.add_argument(
        "--test-num-users",
        default="1,2,3,4,5",
        help="Comma-separated user counts allowed in test. Defaults to --num-users.",
=======
        default=None,
        help="Comma-separated user counts allowed in train. Default: 1,2,3.",
    )
    parser.add_argument(
        "--test-num-users",
        default=None,
        help="Comma-separated user counts allowed in test. Default: 1,2,3,4,5.",
>>>>>>> 588b05913c5a26cb8ad402185084b33bb5918d6a
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
        help="Train rows to sample for GeneralAuthority.",
    )
    parser.add_argument(
        "--general-test-rows",
        type=int,
        default=1500,
        help="Test rows to sample for GeneralAuthority.",
    )
    parser.add_argument(
        "--tool-train-rows",
        type=int,
        default=1000,
        help="Train rows to sample for ToolAuthority.",
    )
    parser.add_argument(
        "--tool-test-rows",
        type=int,
        default=3000,
        help="Test rows to sample for ToolAuthority.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--base-output-dir", type=Path, default=DEFAULT_BASE_OUTPUT_DIR)
    parser.add_argument(
        "--paraphrase-output-dir",
        type=Path,
        default=DEFAULT_PARAPHRASE_OUTPUT_DIR,
    )
    parser.add_argument("--paraphrase-version", default="synthetic_v1")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--num-table-examples", type=int, default=5)
    parser.add_argument("--num-full-examples", type=int, default=1)
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

    print("[1] Build base authority data")
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
            "GeneralAuthority": args.general_train_rows,
            "ToolAuthority": args.tool_train_rows,
        },
        test_rows_by_dataset={
            "GeneralAuthority": args.general_test_rows,
            "ToolAuthority": args.tool_test_rows,
        },
        seed=args.seed,
    )
    print(json.dumps(base_counts, indent=2, ensure_ascii=False))
    print()

    print("[2] Build synthetic paraphrase data")
    paraphrase_counts = make_paraphrase_data(
        input_dir=args.base_output_dir,
        output_dir=args.paraphrase_output_dir,
        paraphrase_version=args.paraphrase_version,
        prompt_version=args.prompt_version,
    )
    print(json.dumps(paraphrase_counts, indent=2, ensure_ascii=False))
    print()

    print("[3] Preview final data")
    for dataset_name in DATASET_NAMES:
        if dataset_name not in paraphrase_counts:
            continue
        path = args.paraphrase_output_dir / dataset_name / "train.jsonl"
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
