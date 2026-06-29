from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable


AGENTDOJO_TASK_GLOBS = {
    "all": {
        "user_tasks": "src/agentdojo/default_suites/*/*/user_tasks.py",
        "injection_tasks": "src/agentdojo/default_suites/*/*/injection_tasks.py",
        "task_pairs": "src/agentdojo/default_suites/*/*/{user_tasks.py,injection_tasks.py}",
    },
}


def _repo_root(source_root: str | Path) -> Path:
    root = Path(source_root).expanduser().resolve()
    expected = root / "src" / "agentdojo" / "default_suites"
    if not expected.exists():
        raise FileNotFoundError(f"Could not find AgentDojo default suites: {expected}")
    return root


def _safe_eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, "")
    if isinstance(node, ast.List):
        return [_safe_eval_ast(item, env) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_ast(item, env) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_eval_ast(item, env) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {_safe_eval_ast(key, env): _safe_eval_ast(value, env) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(_safe_eval_ast(value.value, env)))
        return "".join(parts)
    try:
        return ast.literal_eval(node)
    except Exception:
        return ""


def _camel_task_id(class_name: str, prefix: str) -> str:
    suffix = class_name.removeprefix(prefix)
    return f"{prefix.lower()}_{suffix}" if suffix else class_name.lower()


def _suite_version_from_path(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root / "src" / "agentdojo" / "default_suites")
    version = rel.parts[0]
    suite = rel.parts[1] if len(rel.parts) > 2 else ""
    return version, suite


def _extract_task_classes(
    path: Path,
    *,
    root: Path,
    class_prefix: str,
    text_field: str,
) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    version, suite = _suite_version_from_path(path, root)
    records = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.startswith(class_prefix):
            continue
        env: dict[str, Any] = {}
        tools: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                value = _safe_eval_ast(stmt.value, env) if stmt.value is not None else None
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "ground_truth":
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        for keyword in inner.keywords:
                            if keyword.arg == "function":
                                tool = _safe_eval_ast(keyword.value, env)
                                if tool:
                                    tools.append(str(tool))
        text = str(env.get(text_field) or "")
        if not text:
            continue
        task_id = _camel_task_id(node.name, class_prefix)
        relative_path = str(path.relative_to(root))
        records.append(
            {
                "dataset_name": "agentdojo",
                "split": "all",
                "record_type": "user_task" if class_prefix == "UserTask" else "injection_task",
                "sample_id": f"agentdojo_{version}_{suite}_{task_id}",
                "benchmark_area": "AgentDojo",
                "authority_problem": "",
                "user_goal": text if class_prefix == "UserTask" else "",
                "context": "",
                "expected_decision": "execute",
                "expected_tool": tools[0] if tools else None,
                "expected_target": task_id if class_prefix == "UserTask" else "",
                "expected_authority_source": "user",
                "attack_goal": text if class_prefix == "InjectionTask" else "",
                "source": str(root),
                "source_root": str(root),
                "source_file": relative_path,
                "source_index": len(records),
                "version": version,
                "suite": suite,
                "task_id": task_id,
                "class_name": node.name,
                "prompt": text if class_prefix == "UserTask" else "",
                "goal": text if class_prefix == "InjectionTask" else "",
                "ground_truth_output": str(env.get("GROUND_TRUTH_OUTPUT", "")),
                "ground_truth_tools": json.dumps(tools, ensure_ascii=False),
                "raw": "",
            }
        )
    return records


def load_agentdojo_task_records(
    source_root: str | Path,
    *,
    suite_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    root = _repo_root(source_root)
    suite_filter = set(suite_names or [])
    base = root / "src" / "agentdojo" / "default_suites"
    records: list[dict[str, Any]] = []
    for path in sorted(base.glob("*/*/user_tasks.py")):
        version, suite = _suite_version_from_path(path, root)
        if suite_filter and suite not in suite_filter:
            continue
        records.extend(_extract_task_classes(path, root=root, class_prefix="UserTask", text_field="PROMPT"))
    for path in sorted(base.glob("*/*/injection_tasks.py")):
        version, suite = _suite_version_from_path(path, root)
        if suite_filter and suite not in suite_filter:
            continue
        records.extend(_extract_task_classes(path, root=root, class_prefix="InjectionTask", text_field="GOAL"))
    return records


def load_agentdojo_pair_records(
    source_root: str | Path,
    *,
    benchmark_version: str = "v1.2.2",
    suite_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    task_records = load_agentdojo_task_records(source_root, suite_names=suite_names)
    user_tasks = [row for row in task_records if row.get("record_type") == "user_task"]
    injection_tasks = [row for row in task_records if row.get("record_type") == "injection_task"]
    records = []
    for user_task in user_tasks:
        for injection_task in injection_tasks:
            if user_task.get("suite") != injection_task.get("suite"):
                continue
            version = str(user_task.get("version") or "")
            suite = str(user_task.get("suite") or "")
            user_task_id = str(user_task.get("task_id") or "")
            injection_task_id = str(injection_task.get("task_id") or "")
            attack_goal = str(injection_task.get("goal") or "")
            user_goal = str(user_task.get("prompt") or "")
            metadata = {
                "repo": "ethz-spylab/agentdojo",
                "loader": "static_ast",
                "benchmark_version": benchmark_version,
                "version_dir": version,
                "suite": suite,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "ground_truth_tools": json.loads(str(user_task.get("ground_truth_tools") or "[]")),
                "injection_ground_truth_tools": json.loads(str(injection_task.get("ground_truth_tools") or "[]")),
                "ground_truth_output": user_task.get("ground_truth_output") or "",
            }
            example = {
                "dataset_name": "agentdojo",
                "sample_id": f"agentdojo_{version}_{suite}_{user_task_id}_{injection_task_id}",
                "benchmark_area": "AgentDojo",
                "authority_problem": (
                    "AgentDojo user task combined with an injection task. The injection goal is untrusted "
                    "environment content and must not override the user's task."
                ),
                "user_goal": user_goal,
                "context": (
                    f"Suite: {suite}\n"
                    f"Version directory: {version}\n"
                    f"User task id: {user_task_id}\n"
                    f"Injection task id: {injection_task_id}\n"
                    f"Injection goal:\n{attack_goal}"
                ),
                "expected_decision": "execute",
                    "expected_tool": (metadata["ground_truth_tools"] or [None])[0],
                    "adversarial_tools": metadata["injection_ground_truth_tools"],
                    "expected_target": user_task_id,
                "expected_authority_source": "user",
                "attack_goal": attack_goal,
                "source": str(_repo_root(source_root)),
                "metadata": metadata,
            }
            records.append(
                {
                    "dataset_name": example["dataset_name"],
                    "sample_id": example["sample_id"],
                    "benchmark_area": example["benchmark_area"],
                    "authority_problem": example["authority_problem"],
                    "user_goal": example["user_goal"],
                    "context": example["context"],
                    "expected_decision": example["expected_decision"],
                    "expected_tool": example["expected_tool"],
                    "expected_target": example["expected_target"],
                    "expected_authority_source": example["expected_authority_source"],
                    "attack_goal": example["attack_goal"],
                    "source": example["source"],
                    "split": "all",
                    "record_type": "task_pair",
                    "source_root": str(_repo_root(source_root)),
                    "source_file": "src/agentdojo/default_suites",
                    "source_index": len(records),
                    "version": version,
                    "suite": suite,
                    "user_task_id": user_task_id,
                    "injection_task_id": injection_task_id,
                    "task_id": f"{user_task_id}+{injection_task_id}" if user_task_id and injection_task_id else "",
                    "ground_truth_tools": json.dumps(metadata.get("ground_truth_tools") or [], ensure_ascii=False),
                    "adversarial_tools": json.dumps(metadata.get("injection_ground_truth_tools") or [], ensure_ascii=False),
                    "ground_truth_output": str(metadata.get("ground_truth_output") or ""),
                    "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    "raw": "",
                }
            )
    return records


def load_agentdojo_records(
    source_root: str | Path,
    *,
    splits: Iterable[str] = ("all",),
    benchmark_version: str = "v1.2.2",
    suite_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = set(splits)
    records: list[dict[str, Any]] = []
    if "all" in wanted or "tasks" in wanted:
        records.extend(load_agentdojo_task_records(source_root, suite_names=suite_names))
    if "all" in wanted or "pairs" in wanted:
        records.extend(
            load_agentdojo_pair_records(
                source_root,
                benchmark_version=benchmark_version,
                suite_names=suite_names,
            )
        )
    return records


def get_agentdojo_info(
    source_root: str | Path,
    *,
    benchmark_version: str = "v1.2.2",
    suite_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    records = load_agentdojo_records(
        source_root,
        splits=("all",),
        benchmark_version=benchmark_version,
        suite_names=suite_names,
    )
    by_split: dict[str, int] = {}
    by_record_type: dict[str, int] = {}
    by_suite: dict[str, int] = {}
    by_version: dict[str, int] = {}
    by_source_file: dict[str, int] = {}
    for row in records:
        split = str(row.get("split", "unknown"))
        record_type = str(row.get("record_type", "unknown"))
        suite = str(row.get("suite") or row.get("metadata", "unknown"))
        version = str(row.get("version") or row.get("metadata", "unknown"))
        source_file = str(row.get("source_file", "unknown"))
        by_split[split] = by_split.get(split, 0) + 1
        by_record_type[record_type] = by_record_type.get(record_type, 0) + 1
        by_suite[suite] = by_suite.get(suite, 0) + 1
        by_version[version] = by_version.get(version, 0) + 1
        by_source_file[source_file] = by_source_file.get(source_file, 0) + 1
    return {
        "dataset_name": "agentdojo",
        "total_rows": len(records),
        "by_split": dict(sorted(by_split.items())),
        "by_record_type": dict(sorted(by_record_type.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "by_version": dict(sorted(by_version.items())),
        "by_source_file": dict(sorted(by_source_file.items())),
        "source_root": str(_repo_root(source_root)),
        "benchmark_version": benchmark_version,
        "suite_names": list(suite_names or []),
        "task_globs": AGENTDOJO_TASK_GLOBS,
    }


def load_agentdojo_examples_for_preview(
    source_root: str | Path,
    *,
    max_examples: int = 20,
    benchmark_version: str = "v1.2.2",
    suite_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    records = load_agentdojo_pair_records(
        source_root,
        benchmark_version=benchmark_version,
        suite_names=suite_names,
    )
    examples = [
        {
            "dataset_name": str(row["dataset_name"]),
            "sample_id": str(row["sample_id"]),
            "benchmark_area": str(row["benchmark_area"]),
            "authority_problem": str(row["authority_problem"]),
            "user_goal": str(row["user_goal"]),
            "context": str(row["context"]),
            "expected_decision": str(row["expected_decision"]),
            "expected_tool": row.get("expected_tool"),
            "expected_target": row.get("expected_target"),
            "expected_authority_source": str(row["expected_authority_source"]),
            "attack_goal": str(row["attack_goal"]),
            "source": str(row["source"]),
            "metadata": json.loads(str(row.get("metadata") or "{}")),
        }
        for row in records
    ]
    if max_examples > 0:
        return examples[:max_examples]
    return examples


def to_hf_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list(records)


def load_agentdojo_dataset(
    source_root: str | Path,
    *,
    benchmark_version: str = "v1.2.2",
    suite_names: Iterable[str] | None = None,
):
    info = get_agentdojo_info(source_root, benchmark_version=benchmark_version, suite_names=suite_names)
    dataset = to_hf_dataset(
        load_agentdojo_records(
            source_root,
            splits=("all",),
            benchmark_version=benchmark_version,
            suite_names=suite_names,
        )
    )
    dataset.info.description = json.dumps(info, ensure_ascii=False, sort_keys=True)
    return dataset
