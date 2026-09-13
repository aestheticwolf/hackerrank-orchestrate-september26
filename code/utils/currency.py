from datetime import date
from typing import Optional

import pandas as pd


SUPPORTED_CURRENCIES = {
    "INR",
    "ZAR",
    "IDR",
    "USD",
    "EUR",
}


def normalize_currency(currency: object) -> str:
    """
    Normalize a currency value to an uppercase three-letter code.
    """

    if currency is None:
        raise ValueError("Currency cannot be None")

    value = str(currency).strip().upper()

    if not value:
        raise ValueError("Currency cannot be empty")

    if value not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {value}")

    return value


def find_rate(
    exchange_rates: pd.DataFrame,
    from_currency: str,
    to_currency: str,
    rate_date: date,
) -> Optional[float]:
    """
    Find the supplied exchange rate for a currency pair on a date.

    The challenge provides dated exchange rates. We only use those
    supplied rates and never query an external exchange-rate service.

    If the currencies are identical, the rate is 1.0.

    Returns None when no applicable supplied rate exists.
    """

    source = normalize_currency(from_currency)
    target = normalize_currency(to_currency)

    if source == target:
        return 1.0

    rates = exchange_rates.copy()

    if rates.empty:
        return None

    rates["rate_date"] = pd.to_datetime(
        rates["rate_date"],
        errors="coerce",
    ).dt.date

    matches = rates[
        (rates["from_currency"].astype(str).str.upper() == source)
        & (rates["to_currency"].astype(str).str.upper() == target)
        & (rates["rate_date"] == rate_date)
    ]

    if not matches.empty:
        return float(matches.iloc[0]["rate"])

    return None


def convert_amount(
    amount: float,
    from_currency: str,
    to_currency: str,
    rate_date: date,
    exchange_rates: pd.DataFrame,
) -> float:
    """
    Convert an amount using the supplied exchange-rate dataset.

    Same-currency conversions return the original amount.

    Raises ValueError when the required supplied exchange rate
    cannot be found.
    """

    source = normalize_currency(from_currency)
    target = normalize_currency(to_currency)

    if source == target:
        return float(amount)

    rate = find_rate(
        exchange_rates=exchange_rates,
        from_currency=source,
        to_currency=target,
        rate_date=rate_date,
    )

    if rate is None:
        raise ValueError(
            f"No supplied exchange rate found for "
            f"{source}->{target} on {rate_date}"
        )

    return float(amount) * rate