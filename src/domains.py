"""Category definitions for synthetic authority data generation."""

from __future__ import annotations

from typing import TypeAlias


CategoryValues: TypeAlias = dict[str, list[str]]


DEFAULT_CATEGORIES: CategoryValues = {
    "day": [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ],
    "date": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    ],
    "time": [
        "00:00-04:59",
        "05:00-08:59",
        "09:00-12:59",
        "13:00-16:59",
        "17:00-20:59",
        "21:00-23:59",
    ],
    "month": [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ],
    "year": [
        "2024",
        "2025",
        "2026",
        "2027",
        "2028",
        "2029",
        "2030",
    ],
}


GENERAL_AUTHORITY_TRAIN_CATEGORIES = ("day", "date", "time")
GENERAL_AUTHORITY_TEST_CATEGORIES = (
    "day",
    "date",
    "time",
    "month",
    "year",
)


# ---------------------------------------------------------------------
TRAIN_TOOL_NAMES = (
    "send_email",
    "send_text_message",
    "submit_online_form",
)

TEST_TOOL_NAMES = (
    "make_phone_call",
    "send_voice_message",
    "start_live_chat",
)

ALL_TOOL_NAMES = TRAIN_TOOL_NAMES + TEST_TOOL_NAMES


# ---------------------------------------------------------------------
# Tool authority categories
#
# Example:
#   tool             = submit_online_form
#   recipient        = family_member
#   information_type = payment_information
#   purpose          = book_hotel
#
# Interpretation:
#   Submit payment information through an online form to a family member
#   to book a hotel.
# ---------------------------------------------------------------------

TOOL_AUTHORITY_CATEGORIES: CategoryValues = {
    # The person or group receiving the tool-mediated action or information.
    "recipient": [
        "self",
        "family_member",
        "friend",
        "coworker",
        "manager",
        "client",
    ],

    # The information transmitted through the selected tool.
    "information_type": [
        "contact_information",
        "identity_information",
        "payment_information",
        "schedule_information",
        "location_information",
        "preference_information",
        "supporting_documents",
        "special_requirements",
        "message_content",
    ],

    # The task that the transmitted information is intended to accomplish.
    "purpose": [
        "book_hotel",
        "book_flight",
        "reserve_restaurant",
        "schedule_appointment",
        "purchase_product",
        "purchase_ticket",
        "submit_application",
        "request_service",
        "request_assistance",
        "provide_update",
    ],
    "day": [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ],
    "time": [
        "00:00-04:59",
        "05:00-08:59",
        "09:00-12:59",
        "13:00-16:59",
        "17:00-20:59",
        "21:00-23:59",
    ],
}


TOOL_AUTHORITY_TRAIN_CATEGORIES = (
    "recipient",
    "information_type",
    "purpose",
)
TOOL_AUTHORITY_TEST_CATEGORIES = (
    "recipient",
    "information_type",
    "purpose",
    "day",
    "time",
)
