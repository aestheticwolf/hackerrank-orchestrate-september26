from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    REQUESTS_FILE,
    FINANCIAL_PROFILES_FILE,
    FINANCIAL_EVENTS_FILE,
    PAYMENT_OPTIONS_FILE,
    EXCHANGE_RATES_FILE,
    MESSAGES_FILE,
    IMAGES_FILE,
)


class Dataset:
    """
    Container for all raw challenge datasets.

    The loader keeps the original CSV data as pandas DataFrames.
    Later layers will transform these records into our domain models.
    """

    def __init__(
        self,
        requests: pd.DataFrame,
        financial_profiles: pd.DataFrame,
        financial_events: pd.DataFrame,
        payment_options: pd.DataFrame,
        exchange_rates: pd.DataFrame,
        messages: pd.DataFrame,
        images: pd.DataFrame,
    ):
        self.requests = requests
        self.financial_profiles = financial_profiles
        self.financial_events = financial_events
        self.payment_options = payment_options
        self.exchange_rates = exchange_rates
        self.messages = messages
        self.images = images


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Read one CSV file and fail with a useful error if it does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path)


def load_dataset() -> Dataset:
    """
    Load all required challenge CSV files.
    """

    return Dataset(
        requests=_read_csv(REQUESTS_FILE),
        financial_profiles=_read_csv(FINANCIAL_PROFILES_FILE),
        financial_events=_read_csv(FINANCIAL_EVENTS_FILE),
        payment_options=_read_csv(PAYMENT_OPTIONS_FILE),
        exchange_rates=_read_csv(EXCHANGE_RATES_FILE),
        messages=_read_csv(MESSAGES_FILE),
        images=_read_csv(IMAGES_FILE),
    )


def get_requests(dataset: Dataset) -> pd.DataFrame:
    """
    Return all purchase requests.
    """

    return dataset.requests.copy()


def get_user_profile(
    dataset: Dataset,
    user_id: str,
) -> Optional[pd.Series]:
    """
    Return the financial profile for one user.
    """

    matches = dataset.financial_profiles[
        dataset.financial_profiles["user_id"].astype(str) == str(user_id)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def get_user_events(
    dataset: Dataset,
    user_id: str,
) -> pd.DataFrame:
    """
    Return all financial events belonging to one user.
    """

    return dataset.financial_events[
        dataset.financial_events["user_id"].astype(str) == str(user_id)
    ].copy()


def get_request_payment_options(
    dataset: Dataset,
    request_id: str,
) -> pd.DataFrame:
    """
    Return all payment options available for one request.
    """

    return dataset.payment_options[
        dataset.payment_options["request_id"].astype(str) == str(request_id)
    ].copy()


def get_user_messages(
    dataset: Dataset,
    user_id: str,
) -> pd.DataFrame:
    """
    Return all messages belonging to one user.
    """

    return dataset.messages[
        dataset.messages["user_id"].astype(str) == str(user_id)
    ].copy()


def get_user_images(
    dataset: Dataset,
    user_id: str,
) -> pd.DataFrame:
    """
    Return all images belonging to one user.
    """

    return dataset.images[
        dataset.images["user_id"].astype(str) == str(user_id)
    ].copy()


def get_exchange_rates(dataset: Dataset) -> pd.DataFrame:
    """
    Return the supplied fixed exchange-rate dataset.

    We intentionally do not call any external exchange-rate API.
    """

    return dataset.exchange_rates.copy()


if __name__ == "__main__":
    dataset = load_dataset()

    print("Dataset loaded successfully.")
    print()

    print("Requests:", len(dataset.requests))
    print("Financial profiles:", len(dataset.financial_profiles))
    print("Financial events:", len(dataset.financial_events))
    print("Payment options:", len(dataset.payment_options))
    print("Exchange rates:", len(dataset.exchange_rates))
    print("Messages:", len(dataset.messages))
    print("Images:", len(dataset.images))