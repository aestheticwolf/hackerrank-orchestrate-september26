from __future__ import annotations

from typing import Optional

import pandas as pd

from data_loader import Dataset
from models import Evidence


def _safe_text(value: object) -> str:
    """
    Convert a value into clean text.

    Missing values are represented as an empty string.
    """

    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "nat"}:
        return ""

    return text


def collect_message_evidence(
    dataset: Dataset,
    user_id: str,
    request_id: Optional[str] = None,
    event_id: Optional[str] = None,
) -> list[Evidence]:
    """
    Collect message records relevant to a user, request, or event.

    No financial facts are inferred here.

    Messages are treated as untrusted evidence and are only collected
    for later extraction and validation.
    """

    messages = dataset.messages.copy()

    messages["user_id"] = messages["user_id"].astype(str)

    matches = messages[messages["user_id"] == str(user_id)]

    if request_id is not None:
        matches = matches[
            matches["request_id"].fillna("").astype(str) == str(request_id)
        ]

    if event_id is not None:
        matches = matches[
            matches["related_event_id"].fillna("").astype(str) == str(event_id)
        ]

    evidence: list[Evidence] = []

    for _, row in matches.iterrows():
        evidence.append(
            Evidence(
                evidence_id=_safe_text(row.get("message_id")),
                user_id=_safe_text(row.get("user_id")),
                request_id=_optional_text(row.get("request_id")),
                event_id=_optional_text(row.get("related_event_id")),
                source_type="message",
                content=_safe_text(row.get("message_text")),
            )
        )

    return evidence


def collect_image_evidence(
    dataset: Dataset,
    user_id: str,
    request_id: Optional[str] = None,
    event_id: Optional[str] = None,
) -> list[Evidence]:
    """
    Collect image records relevant to a user, request, or event.

    This function only records the image relationship.

    Actual image interpretation will happen in the AI extraction layer.
    """

    images = dataset.images.copy()

    images["user_id"] = images["user_id"].astype(str)

    matches = images[images["user_id"] == str(user_id)]

    if request_id is not None:
        matches = matches[
            matches["request_id"].fillna("").astype(str) == str(request_id)
        ]

    if event_id is not None:
        matches = matches[
            matches["related_event_id"].fillna("").astype(str) == str(event_id)
        ]

    evidence: list[Evidence] = []

    for _, row in matches.iterrows():
        image_id = _safe_text(row.get("image_id"))
        related_event_id = _optional_text(row.get("related_event_id"))
        related_request_id = _optional_text(row.get("request_id"))

        evidence.append(
            Evidence(
                evidence_id=image_id,
                user_id=_safe_text(row.get("user_id")),
                request_id=related_request_id,
                event_id=related_event_id,
                source_type="image",
                content=image_id,
            )
        )

    return evidence


def collect_user_evidence(
    dataset: Dataset,
    user_id: str,
) -> list[Evidence]:
    """
    Collect all message and image evidence belonging to a user.
    """

    return (
        collect_message_evidence(dataset, user_id)
        + collect_image_evidence(dataset, user_id)
    )


def collect_request_evidence(
    dataset: Dataset,
    user_id: str,
    request_id: str,
) -> list[Evidence]:
    """
    Collect all message and image evidence attached to a request.
    """

    return (
        collect_message_evidence(
            dataset,
            user_id=user_id,
            request_id=request_id,
        )
        + collect_image_evidence(
            dataset,
            user_id=user_id,
            request_id=request_id,
        )
    )


def collect_event_evidence(
    dataset: Dataset,
    user_id: str,
    event_id: str,
) -> list[Evidence]:
    """
    Collect all message and image evidence attached to a financial event.
    """

    return (
        collect_message_evidence(
            dataset,
            user_id=user_id,
            event_id=event_id,
        )
        + collect_image_evidence(
            dataset,
            user_id=user_id,
            event_id=event_id,
        )
    )


def _optional_text(value: object) -> Optional[str]:
    """
    Convert a possibly missing CSV value into Optional[str].
    """

    text = _safe_text(value)

    if not text:
        return None

    return text


def find_blank_amount_events(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Return financial events whose amount is missing.

    Missing amounts must remain unresolved until supported evidence,
    such as an attached image, can provide the amount.

    A blank amount is never treated as zero.
    """

    events = dataset.financial_events.copy()

    return events[events["amount"].isna()].copy()


def find_events_with_image_evidence(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Return financial events that have an image linked through
    images.csv.related_event_id.
    """

    blank_amount_events = find_blank_amount_events(dataset)

    if blank_amount_events.empty or dataset.images.empty:
        return blank_amount_events.iloc[0:0].copy()

    image_event_ids = (
        dataset.images["related_event_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    return blank_amount_events[
        blank_amount_events["event_id"].astype(str).isin(image_event_ids)
    ].copy()


if __name__ == "__main__":
    from data_loader import load_dataset

    dataset = load_dataset()

    blank_events = find_blank_amount_events(dataset)
    image_events = find_events_with_image_evidence(dataset)

    print("Evidence layer loaded successfully.")
    print("Messages:", len(dataset.messages))
    print("Images:", len(dataset.images))
    print("Blank amount events:", len(blank_events))
    print("Blank amount events with image evidence:", len(image_events))