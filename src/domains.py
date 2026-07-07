"""Category definitions for synthetic authority data generation.

Top-level keys are mutually exclusive category axes. A sampled category
combination can include at most one category from each axis, so coarse and fine
categories such as ``day_type`` and ``day_name`` are not sampled together.
"""

from __future__ import annotations

from typing import TypeAlias


CategoryValues: TypeAlias = dict[str, dict[str, list[str]]]


DEFAULT_CATEGORIES: CategoryValues = {
    "day": {
        "day_type": ["weekday", "weekend"],
        "day_name": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ],
    },
    "time": {
        "meridiem": ["AM", "PM"],
        "time_range": [
            "00:00-04:59",
            "05:00-08:59",
            "09:00-12:59",
            "13:00-16:59",
            "17:00-20:59",
            "21:00-23:59",
        ],
    },
}
