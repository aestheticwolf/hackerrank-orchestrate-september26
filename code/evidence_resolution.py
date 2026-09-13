from __future__ import annotations

import pandas as pd

from code.ai.extractor import EvidenceExtractionError, EvidenceExtractor
from code.ai.gemini_provider import GeminiProvider
from code.ai.prompts import (
    EVIDENCE_EXTRACTION_SCHEMA,
    IMAGE_EXTRACTION_SYSTEM_PROMPT,
    build_image_extraction_prompt,
)
from code.data_loader import Dataset
from code.evidence import find_events_with_image_evidence, get_image_path


def _image_id_for_event(dataset: Dataset, event_id: str) -> str | None:
    """
    Find the image_id linked to one financial event.
    """
    matches = dataset.images[
        dataset.images["related_event_id"].astype(str) == str(event_id)
    ]

    if matches.empty:
        return None

    return str(matches.iloc[0]["image_id"])


def _image_path_for_id(dataset: Dataset, image_id: str):
    """
    Resolve an image_id to the actual image file.
    """
    try:
        return get_image_path(image_id)
    except FileNotFoundError as exc:
        raise EvidenceExtractionError(
            f"Image file does not exist for image_id={image_id}"
        ) from exc


def resolve_blank_event_amounts(
    dataset: Dataset,
    extractor: EvidenceExtractor,
) -> pd.DataFrame:
    """
    Resolve missing financial event amounts using linked image evidence.

    Returns a copy of financial_events.csv with successfully
    validated amounts filled in.

    The original dataset is never modified.
    """
    events = dataset.financial_events.copy()

    events_with_images = find_events_with_image_evidence(dataset)

    if events_with_images.empty:
        return events

    for _, row in events_with_images.iterrows():
        event_id = str(row["event_id"])

        image_id = _image_id_for_event(
            dataset,
            event_id,
        )

        if image_id is None:
            continue

        image_path = _image_path_for_id(
            dataset,
            image_id,
        )

        prompt = build_image_extraction_prompt(
    image_id=image_id,
    event_id=event_id,
    event_description=str(row["description"]),
    event_type=str(row["event_type"]),
    category=str(row["category"]),
    event_currency=str(row["currency"]),
)

        evidence = extractor.extract_image(
            image_path=image_path,
            prompt=prompt,
            system_prompt=IMAGE_EXTRACTION_SYSTEM_PROMPT,
            schema=EVIDENCE_EXTRACTION_SCHEMA,
            evidence_id=f"evidence_{image_id}",
            user_id=str(row["user_id"]),
            request_id=None,
            event_id=event_id,
        )

        amount_known = evidence.extracted_facts.get("amount_known", False)
        amount = evidence.extracted_facts.get("amount")

        if not amount_known:
          raise EvidenceExtractionError(
            f"Could not reliably resolve amount for event {event_id} "
            f"from image {image_id}."
          )

        if amount is None:
          raise EvidenceExtractionError(
            f"Event {event_id} was marked amount_known=true "
            "but no amount was returned."
    )

        event_mask = events["event_id"].astype(str) == event_id

        events.loc[event_mask, "amount"] = float(amount)

    return events


def resolve_dataset_event_amounts(
    dataset: Dataset,
    provider: GeminiProvider | None = None,
) -> pd.DataFrame:
    """
    Resolve all currently unresolved event amounts using Gemini.
    """
    if provider is None:
        provider = GeminiProvider()

    extractor = EvidenceExtractor(provider)

    return resolve_blank_event_amounts(
        dataset=dataset,
        extractor=extractor,
    )


if __name__ == "__main__":
    from code.data_loader import load_dataset

    dataset = load_dataset()

    resolved = resolve_dataset_event_amounts(dataset)

    remaining = resolved["amount"].isna().sum()

    print("Evidence resolution completed.")
    print("Total events:", len(resolved))
    print("Remaining unknown amounts:", remaining)