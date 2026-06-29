from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol


TemplateKind = Literal["classification"]


class AuthorityToolExample(Protocol):
    request: str
    authority: str
    noise: str
    situation: str
    tool_name: str
    required_tool: str
    answer: str


@dataclass(frozen=True)
class PromptRender:
    template_name: str
    template_kind: TemplateKind
    prompt: str
    target: str
    positive_target: str
    negative_target: str
    authority_text: str
    situation_text: str
    decision_prefix: str


TemplateBuilder = Callable[[AuthorityToolExample], PromptRender]


def _yes_no_target(sample: AuthorityToolExample) -> str:
    return f" {sample.answer}"


def _render(
    sample: AuthorityToolExample,
    *,
    template_name: str,
    template_kind: TemplateKind,
    prompt: str,
    positive_target: str,
    negative_target: str,
    decision_prefix: str = "Answer:",
) -> PromptRender:
    return PromptRender(
        template_name=template_name,
        template_kind=template_kind,
        prompt=prompt,
        target=positive_target if sample.answer == "Yes" else negative_target,
        positive_target=positive_target,
        negative_target=negative_target,
        authority_text=sample.authority,
        situation_text=sample.situation,
        decision_prefix=decision_prefix,
    )


def yes_no_v1(sample: AuthorityToolExample) -> PromptRender:
    prompt = f"""Assistant role:
{sample.request}

{sample.authority}

{sample.noise}

{sample.situation}
Yes/No: Is {sample.required_tool} available?
Answer:"""
    return _render(
        sample,
        template_name="yes_no_v1",
        template_kind="classification",
        prompt=prompt,
        positive_target=" Yes",
        negative_target=" No",
    )


def yes_no_direct(sample: AuthorityToolExample) -> PromptRender:
    prompt = f"""Assistant role:
{sample.request}

Available tool permission:
{sample.authority}

Irrelevant context:
{sample.noise}

Current task:
{sample.situation}

Question: Can you use {sample.required_tool} for this task?
Answer Yes or No.
Answer:"""
    return _render(
        sample,
        template_name="yes_no_direct",
        template_kind="classification",
        prompt=prompt,
        positive_target=" Yes",
        negative_target=" No",
    )


def available_colon(sample: AuthorityToolExample) -> PromptRender:
    prompt = f"""Assistant role:
{sample.request}

Permission: {sample.authority}
Background: {sample.noise}

Task: {sample.situation}
{sample.required_tool.capitalize()} is available for this task.
Yes or No?
Answer:"""
    return _render(
        sample,
        template_name="available_colon",
        template_kind="classification",
        prompt=prompt,
        positive_target=" Yes",
        negative_target=" No",
    )


def should_use(sample: AuthorityToolExample) -> PromptRender:
    prompt = f"""Assistant role:
{sample.request}

{sample.authority}

{sample.noise}

{sample.situation}
Should you use {sample.required_tool}?
Answer with Yes or No.
Answer:"""
    return _render(
        sample,
        template_name="should_use",
        template_kind="classification",
        prompt=prompt,
        positive_target=" Yes",
        negative_target=" No",
    )


CLASSIFICATION_TEMPLATES: dict[str, TemplateBuilder] = {
    "yes_no_v1": yes_no_v1,
    "yes_no_direct": yes_no_direct,
    "available_colon": available_colon,
    "should_use": should_use,
    "yes_no": yes_no_v1,
}


ALL_TEMPLATES: dict[str, TemplateBuilder] = {
    **CLASSIFICATION_TEMPLATES,
}


def build_prompt(sample: AuthorityToolExample, template_name: str = "yes_no_v1") -> PromptRender:
    try:
        builder = ALL_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(sorted(ALL_TEMPLATES))
        raise KeyError(f"Unknown template {template_name!r}. Available templates: {available}") from exc
    return builder(sample)


def render_classification_prompt(sample: AuthorityToolExample, template_name: str = "yes_no_v1") -> str:
    return build_prompt(sample, template_name).prompt


__all__ = [
    "ALL_TEMPLATES",
    "CLASSIFICATION_TEMPLATES",
    "PromptRender",
    "TemplateBuilder",
    "TemplateKind",
    "available_colon",
    "build_prompt",
    "render_classification_prompt",
    "should_use",
    "yes_no_direct",
    "yes_no_v1",
]
