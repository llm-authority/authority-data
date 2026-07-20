"""Push authority datasets to the Hugging Face Hub.

Run ``make_data.py`` first. For V2 configs, also run ``paraphrase_data.py`` so
cached LLM query paraphrases are available. V1 and V3 configs are pushed from
base rows directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "leo-bjpark/authority"
DEFAULT_BASE_DIR = REPO_ROOT / "data" / "base"
DEFAULT_PARAPHRASE_VERSION = "v2"
DEFAULT_CACHE_PATH = (
    REPO_ROOT / "data" / "paraphrase_cache" / f"{DEFAULT_PARAPHRASE_VERSION}.jsonl"
)
DEFAULT_CONFIGS = (
    "GeneralAuthorityV1",
    "GeneralAuthorityV2",
    "GeneralAuthorityV3",
    "GeneralAuthorityV4",
    "GeneralAuthorityV5",
    "ToolAuthorityV1",
    "ToolAuthorityV2",
    "ToolAuthorityV3",
    "ToolAuthorityV4",
    "ToolAuthorityV5",
)
DEFAULT_OLD_REMOTE_CONFIGS = (
    "GeneralAuthority",
    "ToolAuthority",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
)
DEFAULT_SPLITS = ("train", "test")

from paraphrase_data import (
    cache_key,
    cache_row_evaluation_score,
    input_text_for_row,
    load_paraphrase_cache,
    render_text_with_natural_query,
)
from src.make_paraphrase_data import render_synthetic_paraphrase


@dataclass(frozen=True)
class ConfigSpec:
    config_name: str
    source_config_name: str
    version: str
    use_cache: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push generated authority data to Hugging Face as one dataset repo "
            "with authority configs."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=(
            "Directory containing base config/split JSONL files. "
            f"Default: {DEFAULT_BASE_DIR}"
        ),
    )
    parser.add_argument("--paraphrase-version", default=DEFAULT_PARAPHRASE_VERSION)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Paraphrase cache JSONL. Default: {DEFAULT_CACHE_PATH}",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help=(
            "HF config names to push. Default: GeneralAuthorityV1 "
            "GeneralAuthorityV2 GeneralAuthorityV3 GeneralAuthorityV4 "
            "GeneralAuthorityV5 ToolAuthorityV1 ToolAuthorityV2 ToolAuthorityV3 "
            "ToolAuthorityV4 ToolAuthorityV5."
        ),
    )
    parser.add_argument(
        "--source-configs",
        nargs="+",
        default=None,
        help=(
            "Base directory names aligned with --configs. "
            "Default: infer V2 configs from their matching V1 base directory; "
            "V1, V3, V4, and V5 configs use same-named base directories."
        ),
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
    parser.add_argument(
        "--skip-dataset-card",
        action="store_true",
        help="Do not upload README.md metadata for the Hugging Face dataset viewer.",
    )
    parser.add_argument(
        "--dataset-card-only",
        action="store_true",
        help="Only upload README.md dataset viewer metadata; do not push data files.",
    )
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="For V2 configs, fail if any base row is missing from the paraphrase cache.",
    )
    parser.add_argument(
        "--delete-remote-configs",
        nargs="*",
        default=None,
        help=(
            "Delete old remote config folders before pushing. If passed without "
            "names, deletes GeneralAuthority, ToolAuthority, V1, V2, and V3."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def split_path(base_dir: Path, source_config_name: str, split_name: str) -> Path:
    return base_dir / source_config_name / f"{split_name}.jsonl"


def make_base_push_row(base_row: dict[str, Any], *, version: str) -> dict[str, Any]:
    meta_data = {
        "query": base_row["Query"],
        "paraphrase_version": version,
    }
    v3_metadata = base_row.get("metadata", {}).get("v3")
    if v3_metadata is not None:
        v3_metadata = dict(v3_metadata)
        user_count = v3_metadata.get("user_count")
        if user_count is None:
            user_count = (
                base_row.get("metadata", {}).get("source", {}).get("user_count")
            )
        if user_count is None:
            user_count = len(base_row["AuthoritySetting"]["users"])
        v3_metadata["user_count"] = user_count
        v3_metadata["lower_priority_user_count"] = v3_metadata.get(
            "lower_priority_user_count",
            user_count - 1,
        )
        meta_data["v3"] = v3_metadata
    v4_metadata = base_row.get("metadata", {}).get("v4")
    if v4_metadata is not None:
        meta_data["v4"] = dict(v4_metadata)
    v5_metadata = base_row.get("metadata", {}).get("v5")
    if v5_metadata is not None:
        meta_data["v5"] = dict(v5_metadata)
    return {
        "text": render_synthetic_paraphrase(base_row),
        "label": base_row["Label"],
        "meta_data": meta_data,
    }


def make_cached_push_row(
    base_row: dict[str, Any],
    *,
    cache: dict[tuple[str, str], dict[str, Any]],
    paraphrase_version: str,
) -> dict[str, Any] | None:
    input_text = input_text_for_row(base_row)
    key = cache_key(input_text, paraphrase_version)
    if key not in cache:
        return None

    cache_row = cache[key]
    generated_text = cache_row.get("generated_text") or cache_row["natural_query"]
    score = cache_row_evaluation_score(cache_row)
    return {
        "text": render_text_with_natural_query(base_row, generated_text),
        "label": base_row["Label"],
        "meta_data": {
            "query": base_row["Query"],
            "paraphrase_version": paraphrase_version,
            "query_meaning_score": score,
        },
    }


@dataclass(frozen=True)
class BuildRowsResult:
    rows: list[dict[str, Any]]
    skipped_missing_cache: int
    skipped_eval_zero: int


def build_push_rows(
    *,
    base_dir: Path,
    config_spec: ConfigSpec,
    split_name: str,
    cache: dict[tuple[str, str], dict[str, Any]],
    paraphrase_version: str,
) -> BuildRowsResult:
    path = split_path(base_dir, config_spec.source_config_name, split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a JSONL file, got: {path}")

    rows = []
    skipped_missing_cache = 0
    skipped_eval_zero = 0
    for base_row in read_jsonl(path):
        if config_spec.use_cache:
            push_row = make_cached_push_row(
                base_row,
                cache=cache,
                paraphrase_version=paraphrase_version,
            )
            if push_row is None:
                skipped_missing_cache += 1
                continue
            if push_row["meta_data"]["query_meaning_score"] == 0:
                skipped_eval_zero += 1
                continue
        else:
            push_row = make_base_push_row(base_row, version=config_spec.version)
        rows.append(push_row)
    return BuildRowsResult(
        rows=rows,
        skipped_missing_cache=skipped_missing_cache,
        skipped_eval_zero=skipped_eval_zero,
    )


def collect_query_schema(
    *,
    base_dir: Path,
    config_spec: ConfigSpec,
    splits: list[str],
) -> tuple[list[str], bool]:
    attribute_keys: set[str] = set()
    include_tool = False
    for split_name in splits:
        path = split_path(base_dir, config_spec.source_config_name, split_name)
        for base_row in read_jsonl(path):
            query = base_row["Query"]
            attribute_keys.update(query.get("attributes", {}).keys())
            include_tool = include_tool or "tool" in query
    return sorted(attribute_keys), include_tool


def normalize_query_schema(
    row: dict[str, Any],
    *,
    attribute_keys: list[str],
    include_tool: bool,
) -> dict[str, Any]:
    row = copy.deepcopy(row)
    query = row["meta_data"]["query"]
    attributes = query.get("attributes", {})
    query["attributes"] = {
        attribute_key: attributes.get(attribute_key) for attribute_key in attribute_keys
    }
    if include_tool:
        query["tool"] = query.get("tool")
    else:
        query.pop("tool", None)
    return row


def validate_local_data(
    *,
    base_dir: Path,
    config_specs: list[ConfigSpec],
    splits: list[str],
    cache: dict[tuple[str, str], dict[str, Any]],
    paraphrase_version: str,
    strict_cache: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for config_spec in config_specs:
        summary[config_spec.config_name] = {}
        for split_name in splits:
            result = build_push_rows(
                base_dir=base_dir,
                config_spec=config_spec,
                split_name=split_name,
                cache=cache,
                paraphrase_version=paraphrase_version,
            )
            if config_spec.use_cache and strict_cache and result.skipped_missing_cache:
                raise KeyError(
                    f"{config_spec.config_name}/{split_name} is missing "
                    f"{result.skipped_missing_cache} paraphrase cache entries."
                )
            summary[config_spec.config_name][split_name] = {
                "source_config": config_spec.source_config_name,
                "version": config_spec.version,
                "use_cache": config_spec.use_cache,
                "path": str(
                    split_path(base_dir, config_spec.source_config_name, split_name)
                ),
                "rows": len(result.rows),
                "skipped_missing_cache": result.skipped_missing_cache,
                "skipped_eval_zero": result.skipped_eval_zero,
                "columns": list(result.rows[0].keys()) if result.rows else [],
            }
    return summary


def build_dataset_dict(
    *,
    base_dir: Path,
    config_spec: ConfigSpec,
    splits: list[str],
    cache: dict[tuple[str, str], dict[str, Any]],
    paraphrase_version: str,
    strict_cache: bool,
):
    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to push to Hugging Face. "
            "Install it with `pip install -e '.[hf]'` from authority/authority-data."
        ) from exc

    attribute_keys, include_tool = collect_query_schema(
        base_dir=base_dir,
        config_spec=config_spec,
        splits=splits,
    )
    query_features: dict[str, Any] = {
        "attributes": {
            attribute_key: Value("string") for attribute_key in attribute_keys
        },
    }
    if include_tool:
        query_features["tool"] = Value("string")
    meta_features: dict[str, Any] = {
        "query": query_features,
        "paraphrase_version": Value("string"),
    }
    if config_spec.use_cache:
        meta_features["query_meaning_score"] = Value("int64")
    if config_spec.version == "v3":
        meta_features["v3"] = {
            "main_user": Value("string"),
            "user_count": Value("int64"),
            "lower_priority_user_count": Value("int64"),
            "requested_conflict_ratio": Value("float64"),
            "actual_conflict_ratio": Value("float64"),
            "conflict_user_count": Value("int64"),
            "agreeing_lower_priority_user_count": Value("int64"),
        }
    if config_spec.version == "v5":
        meta_features["v5"] = {
            "source_dataset": Value("string"),
            "source_split": Value("string"),
            "source_row_id": Value("int64"),
            "source_user_count": Value("int64"),
            "user_count": Value("int64"),
            "source_rule_count": Value("int64"),
            "rule_count": Value("int64"),
            "rule_count_matches_source": Value("bool"),
            "converted_non_applicable_user_count": Value("int64"),
            "converted_non_applicable_users": [Value("string")],
            "conflict_user_count": Value("int64"),
            "conflict_users": [Value("string")],
            "agreeing_user_count": Value("int64"),
            "agreeing_users": [Value("string")],
        }
    features = Features(
        {
            "text": Value("string"),
            "label": Value("string"),
            "meta_data": meta_features,
        }
    )

    dataset_dict = DatasetDict()
    for split_name in splits:
        result = build_push_rows(
            base_dir=base_dir,
            config_spec=config_spec,
            split_name=split_name,
            cache=cache,
            paraphrase_version=paraphrase_version,
        )
        if config_spec.use_cache and strict_cache and result.skipped_missing_cache:
            raise KeyError(
                f"{config_spec.config_name}/{split_name} is missing "
                f"{result.skipped_missing_cache} paraphrase cache entries."
            )
        rows = result.rows
        if not rows:
            print(
                f"[hf_push]   skipping empty split "
                f"{config_spec.source_config_name}/{split_name}"
            )
            continue
        rows = [
            normalize_query_schema(
                row,
                attribute_keys=attribute_keys,
                include_tool=include_tool,
            )
            for row in rows
        ]
        dataset_dict[split_name] = Dataset.from_list(rows, features=features)
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


def strip_yaml_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :].lstrip()


def build_dataset_card(
    *,
    repo_id: str,
    summary: dict[str, dict[str, dict[str, Any]]],
) -> str:
    lines = [
        "---",
        "configs:",
    ]
    for config_name, split_summary in summary.items():
        lines.append(f"- config_name: {config_name}")
        lines.append("  data_files:")
        for split_name, info in split_summary.items():
            if info["rows"] == 0:
                continue
            lines.append(f"  - split: {split_name}")
            lines.append(f"    path: {config_name}/{split_name}-*.parquet")
    lines.extend(
        [
            "---",
            "",
        ]
    )

    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        body = strip_yaml_front_matter(readme_path.read_text(encoding="utf-8"))
    else:
        body = (
            "# AuthorityBench\n\n"
            "Synthetic authority-decision datasets with deterministic V1 "
            "configs, paraphrased V2 configs, and conflict stress-test V3 "
            "configs.\n"
        )

    return "\n".join(lines) + body.rstrip() + "\n"


def upload_dataset_card(
    *,
    repo_id: str,
    summary: dict[str, dict[str, dict[str, Any]]],
    token: str | None,
    commit_message: str | None,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "The `huggingface_hub` package is required to upload README.md."
        ) from exc

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=build_dataset_card(repo_id=repo_id, summary=summary).encode(
            "utf-8"
        ),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Update dataset card data files metadata",
    )


def print_dataset_card_metadata(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    print("[hf_push] dataset card data_files metadata:")
    for config_name, split_summary in summary.items():
        print(f"[hf_push]   config_name: {config_name}")
        for split_name, info in split_summary.items():
            if info["rows"] == 0:
                continue
            print(
                f"[hf_push]     {split_name}: " f"{config_name}/{split_name}-*.parquet"
            )


def print_summary(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    for config_name, split_summary in summary.items():
        print(f"[hf_push] config = {config_name}")
        for split_name, info in split_summary.items():
            columns = ", ".join(info["columns"])
            cache_text = "cache" if info["use_cache"] else "base"
            print(
                f"[hf_push]   {split_name}: source={info['source_config']}, "
                f"version={info['version']}, mode={cache_text}, "
                f"rows={info['rows']}, "
                f"skipped_missing_cache={info['skipped_missing_cache']}, "
                f"skipped_eval_zero={info['skipped_eval_zero']}, "
                f"columns=[{columns}]"
            )


def config_version(config_name: str) -> str:
    match = re.search(r"V(\d+)$", config_name)
    if not match:
        raise ValueError(
            f"Config name must end with a version suffix like V1, V2, V3, V4, or V5: "
            f"{config_name}"
        )
    return f"v{match.group(1)}"


def default_source_config(config_name: str) -> str:
    version = config_version(config_name)
    if version in {"v1", "v3", "v4", "v5"}:
        return config_name
    return re.sub(r"V\d+$", "V1", config_name)


def resolve_config_specs(
    configs: list[str],
    source_configs: list[str] | None,
) -> list[ConfigSpec]:
    if source_configs is None:
        source_configs = [default_source_config(config_name) for config_name in configs]

    if len(configs) != len(source_configs):
        raise ValueError(
            "--configs and --source-configs must have the same number of entries."
        )
    return [
        ConfigSpec(
            config_name=config_name,
            source_config_name=source_config_name,
            version=config_version(config_name),
            use_cache=config_version(config_name) == "v2",
        )
        for config_name, source_config_name in zip(configs, source_configs)
    ]


def delete_remote_config_folders(
    *,
    repo_id: str,
    config_names: list[str],
    token: str | None,
    dry_run: bool,
) -> None:
    if not config_names:
        config_names = list(DEFAULT_OLD_REMOTE_CONFIGS)

    for config_name in config_names:
        paths = [f"data/{config_name}", config_name]
        for folder_path in paths:
            if dry_run:
                print(
                    "[hf_push] dry-run delete remote folder "
                    f"repo={repo_id} folder={folder_path}"
                )
                continue
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise ImportError(
                    "The `huggingface_hub` package is required to delete "
                    "remote config folders."
                ) from exc

            api = HfApi(token=token)
            try:
                api.delete_folder(
                    repo_id=repo_id,
                    folder_path=folder_path,
                    repo_type="dataset",
                )
                print(f"[hf_push] deleted remote folder: {folder_path}")
            except Exception as exc:
                print(f"[hf_push] could not delete remote folder {folder_path}: {exc}")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    configs = list(args.configs)
    config_specs = resolve_config_specs(configs, args.source_configs)
    splits = list(args.splits)
    cache_path = args.cache_path.resolve()
    needs_cache = any(config_spec.use_cache for config_spec in config_specs)
    cache = load_paraphrase_cache(cache_path) if needs_cache else {}
    paraphrase_version = args.paraphrase_version.lower()

    for config_spec in config_specs:
        if config_spec.use_cache and config_spec.version != paraphrase_version:
            raise ValueError(
                f"{config_spec.config_name} implies {config_spec.version}, "
                f"but --paraphrase-version is {args.paraphrase_version!r}."
            )

    print(f"[hf_push] repo_id = {args.repo_id}")
    print(f"[hf_push] base_dir = {base_dir}")
    print(f"[hf_push] cache_path = {cache_path if needs_cache else 'unused'}")
    print(f"[hf_push] paraphrase_version = {paraphrase_version}")
    print(
        "[hf_push] configs = "
        + ", ".join(
            f"{spec.config_name}<-{spec.source_config_name}" for spec in config_specs
        )
    )
    print(f"[hf_push] splits = {', '.join(splits)}")
    print(f"[hf_push] private = {args.private}")

    if args.delete_remote_configs is not None:
        delete_remote_config_folders(
            repo_id=args.repo_id,
            config_names=list(args.delete_remote_configs),
            token=args.token,
            dry_run=args.dry_run,
        )

    summary = validate_local_data(
        base_dir=base_dir,
        config_specs=config_specs,
        splits=splits,
        cache=cache,
        paraphrase_version=paraphrase_version,
        strict_cache=args.strict_cache,
    )
    print_summary(summary)
    print_dataset_card_metadata(summary)

    if args.dry_run:
        print("[hf_push] dry run complete; no upload performed")
        return

    if args.dataset_card_only:
        if args.skip_dataset_card:
            raise ValueError(
                "--dataset-card-only cannot be used with --skip-dataset-card"
            )
        print("[hf_push] uploading README.md dataset viewer metadata only")
        upload_dataset_card(
            repo_id=args.repo_id,
            summary=summary,
            token=args.token,
            commit_message=args.commit_message,
        )
        print("[hf_push] done")
        return

    for config_spec in config_specs:
        print(
            "[hf_push] building "
            f"`{config_spec.config_name}` from `{config_spec.source_config_name}`"
        )
        dataset_dict = build_dataset_dict(
            base_dir=base_dir,
            config_spec=config_spec,
            splits=splits,
            cache=cache,
            paraphrase_version=paraphrase_version,
            strict_cache=args.strict_cache,
        )
        if not dataset_dict:
            print(
                f"[hf_push] skipping `{config_spec.config_name}` "
                "because it has no rows"
            )
            continue
        print(
            "[hf_push] pushing "
            f"{config_spec.config_name}: "
            + ", ".join(f"{split}={len(dataset_dict[split])}" for split in dataset_dict)
        )
        push_config(
            repo_id=args.repo_id,
            config_name=config_spec.config_name,
            dataset_dict=dataset_dict,
            private=args.private,
            max_shard_size=args.max_shard_size,
            token=args.token,
            commit_message=args.commit_message,
        )

    if args.skip_dataset_card:
        print("[hf_push] skipping dataset card upload")
    else:
        print("[hf_push] uploading README.md dataset viewer metadata")
        upload_dataset_card(
            repo_id=args.repo_id,
            summary=summary,
            token=args.token,
            commit_message=args.commit_message,
        )

    print("[hf_push] done")
    print("[hf_push] example usage:")
    print("  from datasets import load_dataset")
    for config_spec in config_specs:
        print(f"  ds = load_dataset('{args.repo_id}', '{config_spec.config_name}')")
        print("  print(ds)")


if __name__ == "__main__":
    main()
