from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from authority_data._authority import (
    generate_authority_attack_template_data,
    generate_authority_template_data,
)
from authority_data.prompts import render_prompt_v1, render_prompt_v2


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "authority"
SOURCE_AUTHORITY_MODES = ("permission_only", "prohibition_only", "permission_with_prohibition")
AUTHORITY_MODE_NAMES = {
    "permission_only": "permission",
    "prohibition_only": "prohibition",
    "permission_with_prohibition": "permission_and_prohibition",
}
AUTHORITY_MODES = tuple(AUTHORITY_MODE_NAMES.values())
TEST_SPLIT_EVERY_N_SITUATIONS = 4


def _jsonable(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    raise TypeError(f"Cannot serialize row of type {type(row).__name__}")


def _tool_id(tool_name: str) -> str:
    cleaned = tool_name.strip()
    if cleaned.startswith("the "):
        cleaned = cleaned[4:]
    if cleaned.endswith(" tool"):
        cleaned = cleaned[:-5]
    return cleaned.strip()


def _authority_sections(source: dict[str, Any]) -> tuple[str, str]:
    mode = source["authority_specification_mode"]
    permitted = source["allowed_actions"] if mode in {"permission_only", "permission_with_prohibition"} else ""
    prohibited = source["forbidden_actions"] if mode in {"prohibition_only", "permission_with_prohibition"} else ""
    return permitted or "(not specified)", prohibited or "(not specified)"


def _authority_mode(source: dict[str, Any]) -> str:
    return AUTHORITY_MODE_NAMES[str(source["authority_specification_mode"])]


def _prompt_v2(row: dict[str, Any]) -> str:
    return render_prompt_v2(
        permitted=row["permitted"],
        prohibited=row["prohibited"],
        task_goal=row["task_goal"],
        external_context=row["external_context"],
        target_action=row["target_action"],
        intro="You are given an authority specification, a task goal, and a proposed action.",
        task_header="Task goal",
        external_header="Context from external sources",
    )


def _action_type(source_action_type: str) -> str:
    action_type_map = {
        "normal": "authorized",
        "authority_misuse": "misuse",
        "authority_overreach": "overreach",
        "authority_unrelated": "unrelated_tool",
    }
    try:
        return action_type_map[source_action_type]
    except KeyError as exc:
        raise ValueError(f"Unknown action_type: {source_action_type!r}") from exc


def _clean_row(source: dict[str, Any], *, sample_index: int) -> dict[str, Any]:
    action_type = _action_type(source.get("candidate_action_type") or "")
    label = "yes" if action_type == "authorized" else "no"
    permitted, prohibited = _authority_sections(source)
    attack = str(source.get("attack_context", "")).strip()
    noise = str(source["noise"]).replace(attack, "").strip() if attack else str(source["noise"]).strip()
    external_context = "\n\n".join(part for part in (noise, attack) if part)
    row = {
        "sample_id": f"authority_{sample_index:06d}",
        "authority_specification_type": _authority_mode(source),
        "tool": _tool_id(source["tool_name"]),
        "situation_id": source["situation_id"],
        "external_context_id": source["noise_id"],
        "action_type": action_type,
        "permitted": permitted,
        "prohibited": prohibited,
        "task_goal": source["situation"],
        "external_context": external_context,
        "target_action": source["candidate_action"],
        "label": label,
    }
    row["prompt_v1"] = render_prompt_v1(
        permitted=row["permitted"],
        prohibited=row["prohibited"],
        task_goal=row["task_goal"],
        external_context=row["external_context"],
        target_action=row["target_action"],
    )
    row["prompt_v2"] = _prompt_v2(row)
    return row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    situation_ids = sorted({row["situation_id"] for row in rows})
    test_situations = {
        situation_id
        for index, situation_id in enumerate(situation_ids)
        if index % TEST_SPLIT_EVERY_N_SITUATIONS == TEST_SPLIT_EVERY_N_SITUATIONS - 1
    }
    splits = {"train": [], "test": []}
    for row in rows:
        split = "test" if row["situation_id"] in test_situations else "train"
        splits[split].append(row)
    return splits


def _group_rows_by_mode_and_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped = {
        authority_mode: {"train": [], "test": []}
        for authority_mode in AUTHORITY_MODES
    }
    for split, split_rows in _split_rows(rows).items():
        for row in split_rows:
            grouped[row["authority_specification_type"]][split].append(row)
    return grouped


def export_authority_jsonl(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_authorities: int = 20,
    max_situations_per_authority: int = 5,
    max_noise: int = 5,
) -> dict[str, int]:
    source_rows = [
        _jsonable(row)
        for row in generate_authority_template_data(
            template_names=["yes_no_direct"],
            max_authorities=max_authorities,
            max_situations_per_authority=max_situations_per_authority,
            max_noise=max_noise,
            specification_modes=SOURCE_AUTHORITY_MODES,
        )
    ]
    source_rows.extend(
        _jsonable(row)
        for row in generate_authority_attack_template_data(
            label_filter="all",
            max_authorities=max_authorities,
            max_situations_per_authority=max_situations_per_authority,
            max_noise=max_noise,
            specification_modes=SOURCE_AUTHORITY_MODES,
        )
    )
    rows = [_clean_row(row, sample_index=index) for index, row in enumerate(source_rows)]
    rows_by_mode = _group_rows_by_mode_and_split(rows)
    counts = {}
    for authority_mode, rows_by_split in rows_by_mode.items():
        for split, split_rows in rows_by_split.items():
            counts[f"{authority_mode}/{split}"] = write_jsonl(
                output_dir / authority_mode / f"{split}.jsonl",
                split_rows,
            )

    # Remove older exports; canonical files live under authority_mode/train|test.
    for stale_name in (
        "authority.jsonl",
        "train.jsonl",
        "test.jsonl",
        "base.jsonl",
        "attack.jsonl",
        "tool_samples.jsonl",
        "template_samples.jsonl",
        "attack_template_samples.jsonl",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_dir_name in ("permission_only", "prohibition_only"):
        stale_dir = output_dir / stale_dir_name
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export authority_data rows as JSONL files for Hugging Face datasets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-authorities", type=int, default=20)
    parser.add_argument("--max-situations-per-authority", type=int, default=5)
    parser.add_argument("--max-noise", type=int, default=5)
    args = parser.parse_args()

    counts = export_authority_jsonl(
        output_dir=args.output_dir,
        max_authorities=args.max_authorities,
        max_situations_per_authority=args.max_situations_per_authority,
        max_noise=args.max_noise,
    )
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
