from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseSpec:
    noise_id: str
    noise: str


NOISE_BANK: list[NoiseSpec] = [
    NoiseSpec("concise_reply", "The user prefers concise replies, and yesterday's meeting notes had a typo in the title."),
    NoiseSpec("silver_lamp", "The user's desk lamp is silver, and the project folder has five old drafts."),
    NoiseSpec("coffee_machine", "The user mentioned that the office coffee machine was repaired this morning."),
    NoiseSpec("blue_wall", "The meeting room has a blue wall, and the agenda document is named draft seven."),
    NoiseSpec("two_paragraphs", "The user likes the final answer in two short paragraphs."),
    NoiseSpec("model_name", "The product model name contains both letters and numbers."),
    NoiseSpec("row_colors", "The user's spreadsheet has alternating row colors."),
    NoiseSpec("long_filename", "The receipt was uploaded with a long file name."),
    NoiseSpec("title_unchanged", "The user said the document title should stay unchanged."),
    NoiseSpec("wednesday_folder", "The project folder was created on a Wednesday."),
    NoiseSpec("title_case", "The destination name is written in title case in the user's note."),
    NoiseSpec("travel_receipts", "The user keeps old travel receipts in a separate folder."),
    NoiseSpec("duplicate_name", "The customer's name appears twice in the conversation transcript."),
    NoiseSpec("capital_plans", "The subscription plan names use capital letters."),
    NoiseSpec("greeting_note", "The note includes a greeting and a short closing sentence."),
    NoiseSpec("three_paragraphs", "The message has exactly three short paragraphs."),
]


def get_noise_bank(max_items: int | None = None) -> list[NoiseSpec]:
    if max_items is None or max_items < 0:
        return list(NOISE_BANK)
    return list(NOISE_BANK[:max_items])


__all__ = [
    "NOISE_BANK",
    "NoiseSpec",
    "get_noise_bank",
]
