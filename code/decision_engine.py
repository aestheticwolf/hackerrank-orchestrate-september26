from datetime import date
from typing import Optional

from code.forecast import (
    ForecastDay,
    simulate_additional_payment,
    forecast_is_safe,
)


def calculate_safe_payment_today(
    forecast: list[ForecastDay],
    requested_amount: float,
) -> float:
    """
    Find the maximum amount that can be paid today while keeping
    the entire forecast above the required minimum balance.

    The calculation assumes no optional spending changes.
    """

    if requested_amount < 0:
        raise ValueError("requested_amount cannot be negative")

    if not forecast:
        return 0.0

    requested_amount = float(requested_amount)

    if forecast_is_safe(
        simulate_additional_payment(
            forecast,
            forecast[0].forecast_date,
            requested_amount,
        )
    ):
        return requested_amount

    available_above_minimum = (
        forecast[0].starting_balance
        - forecast[0].minimum_balance
    )

    if available_above_minimum <= 0:
        return 0.0

    high = min(
        requested_amount,
        available_above_minimum,
    )

    low = 0.0

    for _ in range(60):
        middle = (low + high) / 2

        simulated = simulate_additional_payment(
            forecast,
            forecast[0].forecast_date,
            middle,
        )

        if forecast_is_safe(simulated):
            low = middle
        else:
            high = middle

    return round(low, 2)


def find_earliest_safe_full_payment_date(
    forecast: list[ForecastDay],
    requested_amount: float,
) -> Optional[date]:
    """
    Find the earliest forecast date on which the full requested amount
    can be paid as one payment while keeping the entire remaining
    forecast safe.
    """

    if requested_amount < 0:
        raise ValueError("requested_amount cannot be negative")

    if not forecast:
        return None

    for day in forecast:
        simulated = simulate_additional_payment(
            forecast,
            day.forecast_date,
            requested_amount,
        )

        if forecast_is_safe(simulated):
            return day.forecast_date

    return None