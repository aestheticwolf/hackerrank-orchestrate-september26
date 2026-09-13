from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from code.models import FinancialEvent
from code.utils.dates import parse_date


@dataclass
class NormalizedCashFlowEvent:
    event_id: str
    user_id: str
    event_type: str
    description: Optional[str]
    category: Optional[str]
    direction: str
    amount: float
    currency: str
    cash_flow_date: date
    original_status: str
    flexibility: Optional[str] = None
    minimum_allowed_amount: Optional[float] = None
    linked_event_id: Optional[str] = None


def normalize_event(event: FinancialEvent) -> Optional[NormalizedCashFlowEvent]:
    """
    Convert one FinancialEvent into a cash-flow event.

    Returns None when the event must not affect cash flow.
    """

    status = str(event.status).strip().lower()
    direction = str(event.direction).strip().lower()

    # Failed and cancelled transactions never affect cash flow.
    if status in {"failed", "cancelled"}:
        return None

    # Unrealized non-cash investment valuations are not cash movements.
    if status == "unrealized" or direction == "non_cash":
        return None

    # Pending credits must not be counted before settlement.
    if status == "pending" and direction == "credit":
        return None

    # Every cash-flow event must have a known amount by this stage.
    if event.amount is None:
        raise ValueError(
            f"Event {event.event_id} has an unknown amount and "
            "must be resolved before normalization."
        )

    if event.amount < 0:
        raise ValueError(
            f"Event {event.event_id} has a negative amount. "
            "Amounts must be stored as positive magnitudes."
        )

    # Settled and scheduled events use their settlement date when available.
    # Pending debits are reserved using the settlement date when available.
    # Otherwise fall back to the event date.
    cash_flow_date = event.settlement_date or event.event_date

    return NormalizedCashFlowEvent(
        event_id=event.event_id,
        user_id=event.user_id,
        event_type=event.event_type,
        description=event.description,
        category=event.category,
        direction=direction,
        amount=float(event.amount),
        currency=event.currency,
        cash_flow_date=cash_flow_date,
        original_status=status,
        flexibility=event.flexibility,
        minimum_allowed_amount=event.minimum_allowed_amount,
        linked_event_id=event.linked_event_id,
    )


def financial_event_from_row(row: pd.Series) -> FinancialEvent:
    """
    Convert one raw financial_events.csv row into our FinancialEvent model.
    """

    def optional_text(value: object) -> Optional[str]:
        if pd.isna(value):
            return None

        text = str(value).strip()

        return text if text else None

    def optional_float(value: object) -> Optional[float]:
        if pd.isna(value):
            return None

        return float(value)

    return FinancialEvent(
        event_id=str(row["event_id"]),
        user_id=str(row["user_id"]),
        event_type=str(row["event_type"]),
        description=optional_text(row["description"]),
        category=optional_text(row["category"]),
        direction=str(row["direction"]),
        amount=optional_float(row["amount"]),
        currency=str(row["currency"]),
        event_date=parse_date(row["event_date"]),
        settlement_date=parse_date(row["settlement_date"]),
        status=str(row["status"]),
        linked_event_id=optional_text(row["linked_event_id"]),
        flexibility=optional_text(row["flexibility"]),
        minimum_allowed_amount=optional_float(row["minimum_allowed_amount"]),
    )


def normalize_events(events: pd.DataFrame) -> list[NormalizedCashFlowEvent]:
    """
    Normalize a raw financial_events DataFrame.

    Events with statuses that do not affect cash flow are omitted.
    Unknown amounts raise an error so they cannot silently become zero.
    """

    normalized: list[NormalizedCashFlowEvent] = []

    for _, row in events.iterrows():
        event = financial_event_from_row(row)
        normalized_event = normalize_event(event)

        if normalized_event is not None:
            normalized.append(normalized_event)

    return normalized


if __name__ == "__main__":
    from code.data_loader import load_dataset

    dataset = load_dataset()
    normalized = normalize_events(dataset.financial_events)

    print("Event normalization loaded successfully.")
    print("Raw events:", len(dataset.financial_events))
    print("Normalized events:", len(normalized))