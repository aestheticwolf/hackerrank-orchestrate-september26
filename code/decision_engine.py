from datetime import date, timedelta
from typing import Optional

from code.forecast import (
    ForecastDay,
    simulate_additional_payment,
    forecast_is_safe,
)
from code.models import PaymentOption, PaymentPlan
from code.utils.dates import installment_dates


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



def evaluate_payment_option(
    forecast: list[ForecastDay],
    option: PaymentOption,
    desired_completion_date: Optional[date],
    max_installment_months: Optional[int],
) -> Optional[PaymentPlan]:
    """
    Evaluate one supplied payment option against the forecast.

    Returns a PaymentPlan only when the exact supplied option is safe
    and satisfies the request constraints.
    """

    if option.payment_method == "full_payment":
        payments = [
            (
                option.first_payment_date,
                option.payment_amount,
            )
        ]

    elif option.payment_method == "installments":
        if (
            option.payment_frequency_days is None
            or option.payment_frequency_days <= 0
        ):
            return None

        payment_dates = installment_dates(
            option.first_payment_date,
            option.number_of_payments,
            option.payment_frequency_days,
        )

        payments = [
            (
                payment_date,
                option.payment_amount,
            )
            for payment_date in payment_dates
        ]

    else:
        return None

    if not payments:
        return None

    completion_date = payments[-1][0]

    if (
        desired_completion_date is not None
        and completion_date > desired_completion_date
    ):
        return None

    if (
        option.payment_method == "installments"
        and max_installment_months is not None
    ):
        maximum_days = max_installment_months * 31

        if (
            completion_date
            > option.first_payment_date
            + timedelta(days=maximum_days)
        ):
            return None

    simulated_forecast = forecast

    for payment_date, payment_amount in payments:
        simulated_forecast = simulate_additional_payment(
            simulated_forecast,
            payment_date,
            payment_amount,
        )

        if not forecast_is_safe(simulated_forecast):
            return None

    return PaymentPlan(
        payment_method=option.payment_method,
        payments=payments,
        total_cost=option.total_payable_amount,
        completion_date=completion_date,
    )