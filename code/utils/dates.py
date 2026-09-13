from datetime import date, timedelta
from typing import Optional


DATE_FORMAT = "%Y-%m-%d"


def parse_date(value: object) -> Optional[date]:
    """
    Convert a YYYY-MM-DD value into a Python date.

    Returns None for blank or missing values.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "nat"}:
        return None

    return date.fromisoformat(text)


def format_date(value: Optional[date]) -> str:
    """
    Convert a Python date into YYYY-MM-DD.

    Returns an empty string when the date is None.
    """

    if value is None:
        return ""

    return value.strftime(DATE_FORMAT)


def date_range(
    start_date: date,
    end_date: date,
) -> list[date]:
    """
    Return every calendar date from start_date through end_date,
    inclusive.
    """

    if end_date < start_date:
        return []

    days = (end_date - start_date).days

    return [
        start_date + timedelta(days=offset)
        for offset in range(days + 1)
    ]


def add_days(
    start_date: date,
    days: int,
) -> date:
    """
    Add a number of calendar days to a date.
    """

    return start_date + timedelta(days=days)


def installment_dates(
    first_payment_date: date,
    number_of_payments: int,
    frequency_days: int,
) -> list[date]:
    """
    Generate payment dates for an installment plan.

    Example:
        first payment = 2025-08-08
        payments = 3
        frequency = 30 days

    Produces:
        2025-08-08
        2025-09-07
        2025-10-07
    """

    if number_of_payments <= 0:
        return []

    if frequency_days < 0:
        raise ValueError("frequency_days cannot be negative")

    return [
        first_payment_date + timedelta(days=index * frequency_days)
        for index in range(number_of_payments)
    ]