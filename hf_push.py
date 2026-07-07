"""Push generated authority JSONL splits to the Hugging Face Hub.

Run ``make_data.py`` first. This script expects files like:

    data/paraphrase/GeneralAuthority/train.jsonl
    data/paraphrase/GeneralAuthority/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "leo-bjpark/authority"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "paraphrase"
DEFAULT_CONFIGS = ("GeneralAuthority",)
DEFAULT_SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push generated authority data to Hugging Face as one dataset repo "
            "with a GeneralAuthority config."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing config/split JSONL files. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Config names to push. Each config must exist under --data-dir.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to push. Each split must exist as <split>.jsonl.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the dataset repo as private.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Shard size passed to DatasetDict.push_to_hub. Default: 500MB.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF login/env credentials.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message for the Hub upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local files and print row counts without uploading.",
    )
    return parser.parse_args()


def read_first_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"JSONL file is empty: {path}")


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def split_path(data_dir: Path, config_name: str, split_name: str) -> Path:
    return data_dir / config_name / f"{split_name}.jsonl"


def validate_split_file(path: Path) -> tuple[int, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a JSONL file, got: {path}")

    first_row = read_first_jsonl_row(path)
    row_count = count_jsonl_rows(path)
    return row_count, list(first_row.keys())


def validate_local_data(
    *,
    data_dir: Path,
    configs: list[str],
    splits: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for config_name in configs:
        summary[config_name] = {}
        for split_name in splits:
            path = split_path(data_dir, config_name, split_name)
            row_count, columns = validate_split_file(path)
            summary[config_name][split_name] = {
                "path": str(path),
                "rows": row_count,
                "columns": columns,
            }
    return summary


def build_dataset_dict(
    *,
    data_dir: Path,
    config_name: str,
    splits: list[str],
):
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to push to Hugging Face. "
            "Install it with `pip install -e '.[hf]'` from authority/authority-data."
        ) from exc

    dataset_dict = DatasetDict()
    for split_name in splits:
        path = split_path(data_dir, config_name, split_name)
        dataset_dict[split_name] = Dataset.from_json(str(path))
    return dataset_dict


def push_config(
    *,
    repo_id: str,
    config_name: str,
    dataset_dict,
    private: bool,
    max_shard_size: str,
    token: str | None,
    commit_message: str | None,
) -> None:
    push_kwargs: dict[str, Any] = {
        "config_name": config_name,
        "private": private,
        "max_shard_size": max_shard_size,
        "token": token,
    }
    if commit_message is not None:
        push_kwargs["commit_message"] = commit_message

    dataset_dict.push_to_hub(repo_id, **push_kwargs)


def print_summary(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    for config_name, split_summary in summary.items():
        print(f"[hf_push] config = {config_name}")
        for split_name, info in split_summary.items():
            columns = ", ".join(info["columns"])
            print(
                f"[hf_push]   {split_name}: rows={info['rows']}, "
                f"columns=[{columns}]"
            )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    configs = list(args.configs)
    splits = list(args.splits)

    print(f"[hf_push] repo_id = {args.repo_id}")
    print(f"[hf_push] data_dir = {data_dir}")
    print(f"[hf_push] configs = {', '.join(configs)}")
    print(f"[hf_push] splits = {', '.join(splits)}")
    print(f"[hf_push] private = {args.private}")

    summary = validate_local_data(
        data_dir=data_dir,
        configs=configs,
        splits=splits,
    )
    print_summary(summary)

    if args.dry_run:
        print("[hf_push] dry run complete; no upload performed")
        return

    for config_name in configs:
        print(f"[hf_push] building `{config_name}`")
        dataset_dict = build_dataset_dict(
            data_dir=data_dir,
            config_name=config_name,
            splits=splits,
        )
        print(
            "[hf_push] pushing "
            f"{config_name}: "
            + ", ".join(f"{split}={len(dataset_dict[split])}" for split in splits)
        )
        push_config(
            repo_id=args.repo_id,
            config_name=config_name,
            dataset_dict=dataset_dict,
            private=args.private,
            max_shard_size=args.max_shard_size,
            token=args.token,
            commit_message=args.commit_message,
        )

    print("[hf_push] done")
    print("[hf_push] example usage:")
    print("  from datasets import load_dataset")
    for config_name in configs:
        print(f"  ds = load_dataset('{args.repo_id}', '{config_name}')")
        print("  print(ds)")


if __name__ == "__main__":
    main()
