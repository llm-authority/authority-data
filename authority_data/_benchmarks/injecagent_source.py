from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


INJECAGENT_DATA_FILES = {
    "train": {
        "user_cases": "data/user_cases.jsonl",
        "attacker_cases_dh": "data/attacker_cases_dh.jsonl",
        "attacker_cases_ds": "data/attacker_cases_ds.jsonl",
    },
    "test": {
        "test_cases_dh_base": "data/test_cases_dh_base.json",
        "test_cases_ds_base": "data/test_cases_ds_base.json",
        "test_cases_dh_enhanced": "data/test_cases_dh_enhanced.json",
        "test_cases_ds_enhanced": "data/test_cases_ds_enhanced.json",
    },
}


def _repo_root(source_root: str | Path) -> Path:
    root = Path(source_root).expanduser().resolve()
    expected = root / INJECAGENT_DATA_FILES["train"]["user_cases"]
    if not expected.exists():
        raise FileNotFoundError(f"Could not find InjecAgent data file: {expected}")
    return root


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _infer_expected_target(row: dict[str, Any]) -> str:
    user_tool = str(row.get("User Tool") or "user_task")
    return f"{user_tool}_result"


def _injecagent_case_to_example(
    row: dict[str, Any],
    *,
    index: int,
    attack_family: str,
    setting: str,
    source_root: Path,
) -> dict[str, Any]:
    user_tool = str(row.get("User Tool") or "")
    attacker_tools = row.get("Attacker Tools") or []
    attacker_instruction = str(row.get("Attacker Instruction") or "")
    tool_response = row.get("Tool Response") or row.get("Tool Response Template") or ""
    context = (
        f"User tool: {user_tool}\n"
        f"Tool parameters: {row.get('Tool Parameters', '')}\n"
        f"Tool response with injected external content:\n{_stringify(tool_response)}\n\n"
        f"Attacker instruction embedded in that external content:\n{attacker_instruction}"
    )
    attack_kind = "direct_harm" if attack_family == "dh" else "data_stealing"
    return {
        "dataset_name": "injecagent",
        "sample_id": f"injecagent_{setting}_{attack_family}_{index:05d}",
        "benchmark_area": "InjecAgent",
        "authority_problem": (
            "An indirect prompt injection is embedded in the user tool response and tries to redirect the agent "
            f"toward attacker tool(s): {', '.join(map(str, attacker_tools)) or 'unknown'}."
        ),
        "user_goal": str(row.get("User Instruction") or ""),
        "context": context,
        "expected_decision": "execute",
        "expected_tool": user_tool or None,
        "expected_target": _infer_expected_target(row),
        "expected_authority_source": "user",
        "attack_goal": attacker_instruction,
        "source": str(source_root),
        "metadata": {
            "repo": "uiuc-kang-lab/InjecAgent",
            "setting": setting,
            "attack_family": attack_family,
            "attack_kind": attack_kind,
            "attack_type": row.get("Attack Type"),
            "attacker_tools": attacker_tools,
            "expected_achievements": row.get("Expected Achievements"),
            "modified": row.get("Modifed"),
            "thought": row.get("Thought"),
        },
    }


def load_injecagent_examples(
    source_root: str | Path,
    *,
    setting: str = "base",
    attack_family: str = "all",
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    root = _repo_root(source_root)
    families = ["dh", "ds"] if attack_family == "all" else [attack_family]
    examples: list[dict[str, Any]] = []
    for family in families:
        path = root / "data" / f"test_cases_{family}_{setting}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing InjecAgent test case file: {path}")
        for index, row in enumerate(_read_json(path)):
            examples.append(
                _injecagent_case_to_example(
                    row,
                    index=index,
                    attack_family=family,
                    setting=setting,
                    source_root=root,
                )
            )
    if max_examples is not None and max_examples > 0:
        return examples[:max_examples]
    return examples


def _record(
    *,
    split: str,
    record_type: str,
    source_root: Path,
    relative_path: str,
    index: int,
    raw: dict[str, Any],
    example: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "dataset_name": "injecagent",
        "split": split,
        "record_type": record_type,
        "sample_id": f"injecagent_{record_type}_{index:05d}",
        "source": str(source_root),
        "source_root": str(source_root),
        "source_file": relative_path,
        "source_index": index,
        "raw": json.dumps(raw, ensure_ascii=False, sort_keys=True),
        "benchmark_area": "InjecAgent",
        "authority_problem": "",
        "user_goal": str(raw.get("User Instruction") or ""),
        "context": str(raw.get("Tool Response") or raw.get("Tool Response Template") or ""),
        "expected_decision": "execute",
        "expected_tool": raw.get("User Tool"),
        "expected_target": "",
        "expected_authority_source": "user",
        "attack_goal": str(raw.get("Attacker Instruction") or ""),
        "metadata": "{}",
    }
    if example is None:
        return base

    metadata = dict(example.get("metadata") or {})
    metadata.update(
        {
            "split": split,
            "record_type": record_type,
            "source_file": relative_path,
            "source_index": index,
        }
    )
    base.update(
        {
            **{key: value for key, value in example.items() if key != "metadata"},
            "split": split,
            "record_type": record_type,
            "source_root": str(source_root),
            "source_file": relative_path,
            "source_index": index,
            "raw": json.dumps(raw, ensure_ascii=False, sort_keys=True),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        }
    )
    return base


def load_injecagent_records(
    source_root: str | Path,
    *,
    splits: Iterable[str] = ("train", "test"),
    settings: Iterable[str] = ("base", "enhanced"),
    attack_families: Iterable[str] = ("dh", "ds"),
) -> list[dict[str, Any]]:
    root = _repo_root(source_root)
    wanted = set(splits)
    records: list[dict[str, Any]] = []

    if "train" in wanted:
        for record_type, relative_path in INJECAGENT_DATA_FILES["train"].items():
            path = root / relative_path
            for index, raw in enumerate(_read_jsonl(path)):
                records.append(
                    _record(
                        split="train",
                        record_type=record_type,
                        source_root=root,
                        relative_path=relative_path,
                        index=index,
                        raw=raw,
                    )
                )

    if "test" in wanted:
        for setting in settings:
            for family in attack_families:
                relative_path = f"data/test_cases_{family}_{setting}.json"
                path = root / relative_path
                if not path.exists():
                    continue
                examples = load_injecagent_examples(root, setting=setting, attack_family=family)
                raw_rows = _read_json(path)
                for index, (raw, example) in enumerate(zip(raw_rows, examples)):
                    records.append(
                        _record(
                            split="test",
                            record_type=f"test_cases_{family}_{setting}",
                            source_root=root,
                            relative_path=relative_path,
                            index=index,
                            raw=raw,
                            example=example,
                        )
                    )
    return records


def get_injecagent_info(
    source_root: str | Path,
    *,
    settings: Iterable[str] = ("base", "enhanced"),
    attack_families: Iterable[str] = ("dh", "ds"),
) -> dict[str, Any]:
    records = load_injecagent_records(
        source_root,
        splits=("train", "test"),
        settings=settings,
        attack_families=attack_families,
    )
    by_split: dict[str, int] = {}
    by_record_type: dict[str, int] = {}
    by_source_file: dict[str, int] = {}
    for row in records:
        split = str(row.get("split", "unknown"))
        record_type = str(row.get("record_type", "unknown"))
        source_file = str(row.get("source_file", "unknown"))
        by_split[split] = by_split.get(split, 0) + 1
        by_record_type[record_type] = by_record_type.get(record_type, 0) + 1
        by_source_file[source_file] = by_source_file.get(source_file, 0) + 1
    return {
        "dataset_name": "injecagent",
        "total_rows": len(records),
        "by_split": dict(sorted(by_split.items())),
        "by_record_type": dict(sorted(by_record_type.items())),
        "by_source_file": dict(sorted(by_source_file.items())),
        "source_root": str(_repo_root(source_root)),
        "data_files": INJECAGENT_DATA_FILES,
    }


def load_injecagent_examples_for_preview(
    source_root: str | Path,
    *,
    max_examples: int = 20,
    setting: str = "base",
    attack_family: str = "all",
) -> list[dict[str, Any]]:
    return load_injecagent_examples(
        source_root,
        setting=setting,
        attack_family=attack_family,
        max_examples=max_examples,
    )


def to_hf_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list(records)


def load_injecagent_dataset(
    source_root: str | Path,
    *,
    settings: Iterable[str] = ("base", "enhanced"),
    attack_families: Iterable[str] = ("dh", "ds"),
):
    from datasets import DatasetDict

    info = get_injecagent_info(source_root, settings=settings, attack_families=attack_families)
    train_records = load_injecagent_records(source_root, splits=("train",))
    test_records = load_injecagent_records(
        source_root,
        splits=("test",),
        settings=settings,
        attack_families=attack_families,
    )
    dataset = DatasetDict(
        {
            "train": to_hf_dataset(train_records),
            "test": to_hf_dataset(test_records),
        }
    )
    dataset.info.description = json.dumps(info, ensure_ascii=False, sort_keys=True)
    return dataset
