"""Benchmark authority prompt transformations for optional comparison runs."""

from .attack_data import (
    AttackTemplateSample,
    generate_attack_decision_samples,
    generate_attack_template_data,
    generate_attack_template_data_from_archive,
    load_agent_adversary_archive,
)

__all__ = [
    "AttackTemplateSample",
    "generate_attack_decision_samples",
    "generate_attack_template_data",
    "generate_attack_template_data_from_archive",
    "load_agent_adversary_archive",
]
