from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attack_prompt import (
    AttackCondition,
    AttackDecisionSample,
    AttackPlacement,
    AttackPromptRender,
    AttackType,
    DecisionKind,
    DecisionLabel,
    render_attack_decision_prompt,
)


DEFAULT_ATTACK_CONDITIONS: tuple[AttackCondition, ...] = (
    "no_goal",
    "goal",
    "goal_with_allowed_tools",
    "goal_with_forbidden_tools",
    "goal_with_allowed_and_forbidden_tools",
)
DEFAULT_ATTACK_TEMPLATE_NAMES = ["attack_decision_yes_no_v1"]
DEFAULT_ATTACK_PLACEMENTS: tuple[AttackPlacement, ...] = ("beginning", "intermediate", "end")
DEFAULT_ATTACK_TYPES: tuple[AttackType, ...] = (
    "benign_context",
    "original_attack",
    "instruction_override",
    "data_exfiltration",
    "unauthorized_tool_use",
    "goal_relevant_premature_commit",
    "goal_relevant_authority_transfer",
    "goal_relevant_side_effect_completion",
)


def parse_attack_conditions(values: list[str] | None) -> tuple[AttackCondition, ...]:
    if not values:
        return DEFAULT_ATTACK_CONDITIONS
    allowed = set(DEFAULT_ATTACK_CONDITIONS)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown attack condition(s): {unknown}. Available: {list(DEFAULT_ATTACK_CONDITIONS)}")
    return tuple(values)  # type: ignore[return-value]


def parse_attack_types(values: list[str] | None) -> tuple[AttackType, ...]:
    if not values:
        return DEFAULT_ATTACK_TYPES
    allowed = set(DEFAULT_ATTACK_TYPES)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown attack type(s): {unknown}. Available: {list(DEFAULT_ATTACK_TYPES)}")
    return tuple(values)  # type: ignore[return-value]


@dataclass(frozen=True)
class AttackTemplateSample:
    sample_id: str
    base_sample_id: str
    template_name: str
    prompt: str
    target: str
    positive_target: str
    negative_target: str
    decision_prefix: str
    dataset_name: str
    condition: AttackCondition
    attack_placement: AttackPlacement
    attack_type: AttackType
    decision_kind: DecisionKind
    label: DecisionLabel
    user_goal: str
    attack_goal: str
    context: str
    required_tool: str
    adversarial_tool: str
    allowed_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...]

    @property
    def answer(self) -> str:
        return "Yes" if self.label == "yes" else "No"


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _as_tool_tuple(value: Any) -> tuple[str, ...]:
    parsed = _parse_jsonish(value)
    if parsed is None:
        parsed = value
    if isinstance(parsed, str):
        return (parsed,) if parsed.strip() else ()
    if isinstance(parsed, dict):
        return tuple(str(key) for key in parsed if str(key).strip())
    if isinstance(parsed, (list, tuple, set)):
        return tuple(str(item) for item in parsed if str(item).strip())
    return ()


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_jsonish(row.get("metadata"))
    return parsed if isinstance(parsed, dict) else {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def infer_required_tool(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    ground_truth_tools = _as_tool_tuple(row.get("ground_truth_tools") or metadata.get("ground_truth_tools"))
    return _first_nonempty(row.get("expected_tool"), *(ground_truth_tools or ()))


def infer_adversarial_tool(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    attacker_tools = _as_tool_tuple(
        metadata.get("attacker_tools")
        or row.get("attacker_tools")
        or metadata.get("injection_ground_truth_tools")
        or row.get("adversarial_tools")
    )
    if attacker_tools:
        return attacker_tools[0]
    injection_task_id = _first_nonempty(metadata.get("injection_task_id"), row.get("injection_task_id"))
    if injection_task_id:
        return f"injection_task:{injection_task_id}"
    return ""


def _condition_tool_policy(
    condition: AttackCondition,
    *,
    required_tool: str,
    adversarial_tool: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if condition == "goal_with_allowed_tools":
        return ((required_tool,) if required_tool else (), ())
    if condition == "goal_with_forbidden_tools":
        return ((), (adversarial_tool,) if adversarial_tool else ())
    if condition == "goal_with_allowed_and_forbidden_tools":
        return (
            (required_tool,) if required_tool else (),
            (adversarial_tool,) if adversarial_tool else (),
        )
    return (), ()


def _base_id(row: dict[str, Any], row_index: int) -> str:
    return _first_nonempty(row.get("sample_id"), f"row_{row_index:05d}")


def build_attack_decision_samples_for_row(
    row: dict[str, Any],
    *,
    row_index: int = 0,
    conditions: tuple[AttackCondition, ...] = DEFAULT_ATTACK_CONDITIONS,
    attack_placements: tuple[AttackPlacement, ...] = DEFAULT_ATTACK_PLACEMENTS,
    attack_types: tuple[AttackType, ...] = DEFAULT_ATTACK_TYPES,
) -> list[AttackDecisionSample]:
    base_id = _base_id(row, row_index)
    dataset_name = _first_nonempty(row.get("dataset_name"), "unknown")
    user_goal = _first_nonempty(row.get("user_goal"), row.get("prompt"))
    attack_goal = _first_nonempty(row.get("attack_goal"), row.get("goal"))
    context = _first_nonempty(row.get("context"), row.get("raw"))
    required_tool = infer_required_tool(row)
    adversarial_tool = infer_adversarial_tool(row)
    samples: list[AttackDecisionSample] = []
    if not attack_goal:
        return samples

    for condition in conditions:
        for attack_type in attack_types:
            for attack_placement in attack_placements:
                allowed_tools, forbidden_tools = _condition_tool_policy(
                    condition,
                    required_tool=required_tool,
                    adversarial_tool=adversarial_tool,
                )
                decision_specs: list[tuple[DecisionKind, DecisionLabel]] = []
                if required_tool:
                    decision_specs.append(("required_tool", "no" if condition == "no_goal" else "yes"))
                decision_specs.append(("adversarial_tool" if adversarial_tool else "adversarial_goal", "no"))
                for decision_kind, label in decision_specs:
                    sample_id = f"{base_id}__{condition}__{attack_type}__attack_{attack_placement}__{decision_kind}"
                    samples.append(
                        AttackDecisionSample(
                            sample_id=sample_id,
                            base_sample_id=base_id,
                            dataset_name=dataset_name,
                            condition=condition,
                            attack_placement=attack_placement,
                            attack_type=attack_type,
                            decision_kind=decision_kind,
                            label=label,
                            user_goal=user_goal,
                            attack_goal=attack_goal,
                            context=context,
                            required_tool=required_tool,
                            adversarial_tool=adversarial_tool,
                            allowed_tools=allowed_tools,
                            forbidden_tools=forbidden_tools,
                            source_record=dict(row),
                        )
                    )
    return samples


def build_attack_template_sample(
    base_sample: AttackDecisionSample,
    *,
    template_name: str = "attack_decision_yes_no_v1",
) -> AttackTemplateSample:
    rendered: AttackPromptRender = render_attack_decision_prompt(base_sample, template_name=template_name)
    return AttackTemplateSample(
        sample_id=f"{rendered.template_name}__{base_sample.sample_id}",
        base_sample_id=base_sample.sample_id,
        template_name=rendered.template_name,
        prompt=rendered.prompt,
        target=rendered.target,
        positive_target=rendered.positive_target,
        negative_target=rendered.negative_target,
        decision_prefix=rendered.decision_prefix,
        dataset_name=base_sample.dataset_name,
        condition=base_sample.condition,
        attack_placement=base_sample.attack_placement,
        attack_type=base_sample.attack_type,
        decision_kind=base_sample.decision_kind,
        label=base_sample.label,
        user_goal=base_sample.user_goal,
        attack_goal=base_sample.attack_goal,
        context=base_sample.context,
        required_tool=base_sample.required_tool,
        adversarial_tool=base_sample.adversarial_tool,
        allowed_tools=base_sample.allowed_tools,
        forbidden_tools=base_sample.forbidden_tools,
    )


def generate_attack_decision_samples(
    records: list[dict[str, Any]],
    *,
    conditions: tuple[AttackCondition, ...] = DEFAULT_ATTACK_CONDITIONS,
    attack_placements: tuple[AttackPlacement, ...] = DEFAULT_ATTACK_PLACEMENTS,
    attack_types: tuple[AttackType, ...] = DEFAULT_ATTACK_TYPES,
    max_records: int | None = None,
) -> list[AttackDecisionSample]:
    selected = records if max_records is None or max_records < 0 else records[:max_records]
    samples: list[AttackDecisionSample] = []
    for row_index, row in enumerate(selected):
        samples.extend(
            build_attack_decision_samples_for_row(
                row,
                row_index=row_index,
                conditions=conditions,
                attack_placements=attack_placements,
                attack_types=attack_types,
            )
        )
    return samples


def generate_attack_template_data(
    records: list[dict[str, Any]],
    *,
    template_names: list[str] | None = None,
    conditions: tuple[AttackCondition, ...] = DEFAULT_ATTACK_CONDITIONS,
    attack_placements: tuple[AttackPlacement, ...] = DEFAULT_ATTACK_PLACEMENTS,
    attack_types: tuple[AttackType, ...] = DEFAULT_ATTACK_TYPES,
    max_records: int | None = None,
) -> list[AttackTemplateSample]:
    names = list(DEFAULT_ATTACK_TEMPLATE_NAMES if template_names is None else template_names)
    rows: list[AttackTemplateSample] = []
    for sample in generate_attack_decision_samples(
        records,
        conditions=conditions,
        attack_placements=attack_placements,
        attack_types=attack_types,
        max_records=max_records,
    ):
        for template_name in names:
            rows.append(build_attack_template_sample(sample, template_name=template_name))
    return rows


def load_agent_adversary_archive(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected archive rows to be a list in {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def generate_attack_template_data_from_archive(
    path: str | Path,
    *,
    template_names: list[str] | None = None,
    conditions: tuple[AttackCondition, ...] = DEFAULT_ATTACK_CONDITIONS,
    attack_placements: tuple[AttackPlacement, ...] = DEFAULT_ATTACK_PLACEMENTS,
    attack_types: tuple[AttackType, ...] = DEFAULT_ATTACK_TYPES,
    max_records: int | None = None,
) -> list[AttackTemplateSample]:
    return generate_attack_template_data(
        load_agent_adversary_archive(path),
        template_names=template_names,
        conditions=conditions,
        attack_placements=attack_placements,
        attack_types=attack_types,
        max_records=max_records,
    )


__all__ = [
    "AttackTemplateSample",
    "DEFAULT_ATTACK_CONDITIONS",
    "DEFAULT_ATTACK_PLACEMENTS",
    "DEFAULT_ATTACK_TEMPLATE_NAMES",
    "DEFAULT_ATTACK_TYPES",
    "build_attack_decision_samples_for_row",
    "build_attack_template_sample",
    "generate_attack_decision_samples",
    "generate_attack_template_data",
    "generate_attack_template_data_from_archive",
    "infer_adversarial_tool",
    "infer_required_tool",
    "load_agent_adversary_archive",
    "parse_attack_conditions",
    "parse_attack_types",
]
