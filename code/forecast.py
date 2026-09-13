from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class RecurringPattern:
    user_id: str
    event_type: str
    category: str
    interval_days: int
    average_amount: float
    minimum_amount: Optional[float]
    flexibility: str
    occurrences: int
    last_date: date


def detect_recurring_patterns(
    financial_events: pd.DataFrame,
    minimum_occurrences: int = 3,
) -> list[RecurringPattern]:
    events = financial_events.copy()

    if events.empty:
        return []

    events["event_date"] = pd.to_datetime(
        events["event_date"],
        errors="coerce",
    )

    events = events[
        (events["status"] == "settled")
        & (events["direction"] == "debit")
        & events["event_date"].notna()
        & events["amount"].notna()
        & events["category"].notna()
    ].copy()

    patterns: list[RecurringPattern] = []

    grouped = events.sort_values("event_date").groupby(
        ["user_id", "event_type", "category"],
        dropna=False,
    )

    for (user_id, event_type, category), group in grouped:
        if len(group) < minimum_occurrences:
            continue

        group = group.sort_values("event_date")

        gaps = (
            group["event_date"]
            .diff()
            .dt.days
            .dropna()
        )

        if gaps.empty:
            continue

        common_gap = int(gaps.mode().iloc[0])

        matching_gaps = (gaps == common_gap).sum()

        if matching_gaps < minimum_occurrences - 1:
            continue

        amounts = pd.to_numeric(
            group["amount"],
            errors="coerce",
        ).dropna()

        if amounts.empty:
            continue

        last_row = group.iloc[-1]

        minimum_amount = None

        if "minimum_allowed_amount" in group.columns:
            minimum_values = pd.to_numeric(
                group["minimum_allowed_amount"],
                errors="coerce",
            ).dropna()

            if not minimum_values.empty:
                minimum_amount = float(minimum_values.iloc[-1])

        flexibility = ""

        if "flexibility" in group.columns:
            flexibility = str(last_row["flexibility"])

        patterns.append(
            RecurringPattern(
                user_id=str(user_id),
                event_type=str(event_type),
                category=str(category),
                interval_days=common_gap,
                average_amount=float(amounts.mean()),
                minimum_amount=minimum_amount,
                flexibility=flexibility,
                occurrences=len(group),
                last_date=last_row["event_date"].date(),
            )
        )

    return patterns


if __name__ == "__main__":
    from code.data_loader import load_dataset

    dataset = load_dataset()

    patterns = detect_recurring_patterns(
        dataset.financial_events
    )

    print(f"Recurring patterns detected: {len(patterns)}")

    for pattern in patterns[:20]:
        print(pattern)