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


@dataclass
class ForecastDay:
    forecast_date: date
    starting_balance: float
    income: float
    expenses: float
    ending_balance: float
    minimum_balance: float

    @property
    def is_safe(self) -> bool:
        return self.ending_balance >= self.minimum_balance


def get_cash_flow_date(event: pd.Series) -> Optional[date]:
    """
    Return the date on which an event should affect available cash.

    Challenge rules:
    - Settled events use settlement_date when available.
    - Scheduled events use settlement_date when available.
    - Pending events are reserved using event_date.
    - Failed and cancelled events have no cash effect.
    - Unrealized events have no cash effect.
    """

    status = str(event.get("status", "")).strip().lower()
    direction = str(event.get("direction", "")).strip().lower()

    if status in {"failed", "cancelled", "unrealized"}:
        return None

    if status in {"settled", "scheduled"}:
        settlement_date = pd.to_datetime(
            event.get("settlement_date"),
            errors="coerce",
        )

        if pd.notna(settlement_date):
            return settlement_date.date()

    event_date = pd.to_datetime(
        event.get("event_date"),
        errors="coerce",
    )

    if pd.isna(event_date):
        return None

    if status == "pending":
        return event_date.date()

    if direction in {"credit", "debit"}:
        return event_date.date()

    return None


def get_cash_flow_amount(event: pd.Series) -> float:
    """
    Return the cash-flow amount for an event.

    Debit events reduce cash.
    Credit events increase cash.

    Unknown amounts are not treated as zero.
    They raise an error so the caller can resolve them
    using available evidence.
    """

    amount = pd.to_numeric(
        event.get("amount"),
        errors="coerce",
    )

    if pd.isna(amount):
        raise ValueError(
            f"Financial event {event.get('event_id')} has an unknown amount"
        )

    direction = str(
        event.get("direction", "")
    ).strip().lower()

    if direction == "debit":
        return -float(amount)

    if direction == "credit":
        return float(amount)

    return 0.0


if __name__ == "__main__":
    from code.data_loader import load_dataset

    dataset = load_dataset()

    patterns = detect_recurring_patterns(
        dataset.financial_events
    )

    print(f"Recurring patterns detected: {len(patterns)}")

    for pattern in patterns[:20]:
        print(pattern)