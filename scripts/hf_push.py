from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "authority_data" / "data"
AUTHORITY_CONFIGS = ("permission", "prohibition", "permission_and_prohibition")
AGENTDOJO_CONFIGS = ("permission", "prohibition", "permission_and_prohibition")
INJECAGENT_CONFIGS = (
    "base_direct_harm",
    "base_data_stealing",
    "enhanced_direct_harm",
    "enhanced_data_stealing",
)


def _load_split_dataset(path: Path):
    return load_dataset(
        "json",
        data_files={
            "train": str(path / "train.jsonl"),
            "test": str(path / "test.jsonl"),
        },
    )


def push_authority_data(
    *,
    repo_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    include_authority: bool = True,
    include_agentdojo: bool = True,
    include_injecagent: bool = True,
) -> None:
    data_dir = data_dir.expanduser().resolve()

    if include_authority:
        authority_dir = data_dir / "authority"
        for config_name in AUTHORITY_CONFIGS:
            dataset_dict = _load_split_dataset(authority_dir / config_name)
            dataset_dict.push_to_hub(repo_id, config_name=config_name)
            print(f"Pushed {config_name} to {repo_id}")

    if include_agentdojo:
        agentdojo_dir = data_dir / "benchmarks" / "agentdojo"
        for config_name in AGENTDOJO_CONFIGS:
            dataset_dict = _load_split_dataset(agentdojo_dir / config_name)
            hub_config_name = f"agentdojo_{config_name}"
            dataset_dict.push_to_hub(repo_id, config_name=hub_config_name)
            print(f"Pushed {hub_config_name} to {repo_id}")

    if include_injecagent:
        injecagent_dir = data_dir / "benchmarks" / "injecagent"
        for config_name in INJECAGENT_CONFIGS:
            dataset_dict = _load_split_dataset(injecagent_dir / config_name)
            hub_config_name = f"injecagent_{config_name}"
            dataset_dict.push_to_hub(repo_id, config_name=hub_config_name)
            print(f"Pushed {hub_config_name} to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push prepared authority_data JSONL files to Hugging Face Hub.")
    parser.add_argument("--repo-id", default="leo-bjpark/authorization_data_v1")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-authority", action="store_true")
    parser.add_argument("--skip-agentdojo", action="store_true")
    parser.add_argument("--skip-injecagent", action="store_true")
    args = parser.parse_args()

    push_authority_data(
        repo_id=args.repo_id,
        data_dir=args.data_dir,
        include_authority=not args.skip_authority,
        include_agentdojo=not args.skip_agentdojo,
        include_injecagent=not args.skip_injecagent,
    )


if __name__ == "__main__":
    main()
