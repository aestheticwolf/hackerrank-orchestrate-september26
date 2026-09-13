from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from code.models import Evidence


class AIProvider(Protocol):
    """
    Interface that any AI provider must implement.

    The provider is responsible only for sending evidence to a model
    and returning the model's raw response.
    """

    def extract_from_image(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        ...

    def extract_from_text(
        self,
        text: str,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        ...


@dataclass
class ExtractedEvidence:
    """
    Validated result returned by the extraction layer.
    """

    amount: float | None
    currency: str | None
    date: str | None
    status: str | None
    description: str | None
    fact_type: str | None
    confidence: float


class EvidenceExtractionError(Exception):
    """Raised when AI evidence cannot be parsed or validated."""


SUPPORTED_CURRENCIES = {"INR", "ZAR", "IDR", "USD", "EUR"}

ALLOWED_FACT_TYPES = {
    "salary",
    "expense",
    "payment",
    "cancellation",
    "settlement",
    "amendment",
    "income",
    "other",
}


def _clean_optional_string(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"null", "none", "nan"}:
        return None

    return text


def _validate_amount(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise EvidenceExtractionError("Amount cannot be boolean")

    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceExtractionError(
            f"Invalid financial amount: {value!r}"
        ) from exc

    if amount < 0:
        raise EvidenceExtractionError(
            f"Financial amount cannot be negative: {amount}"
        )

    return amount


def _validate_currency(value: Any) -> str | None:
    currency = _clean_optional_string(value)

    if currency is None:
        return None

    currency = currency.upper()

    if currency not in SUPPORTED_CURRENCIES:
        raise EvidenceExtractionError(
            f"Unsupported extracted currency: {currency}"
        )

    return currency


def _validate_date(value: Any) -> str | None:
    date_value = _clean_optional_string(value)

    if date_value is None:
        return None

    try:
        from datetime import date

        date.fromisoformat(date_value)
    except ValueError as exc:
        raise EvidenceExtractionError(
            f"Invalid extracted date: {date_value!r}"
        ) from exc

    return date_value


def _validate_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise EvidenceExtractionError(
            "Confidence must be a number between 0 and 1"
        )

    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceExtractionError(
            f"Invalid confidence: {value!r}"
        ) from exc

    if not 0.0 <= confidence <= 1.0:
        raise EvidenceExtractionError(
            f"Confidence must be between 0 and 1: {confidence}"
        )

    return confidence


def parse_extraction_response(raw_response: str) -> ExtractedEvidence:
    """
    Parse and validate the model's JSON response.

    The model is not trusted. Every extracted field is validated before
    it enters the financial decision pipeline.
    """

    if not raw_response or not raw_response.strip():
        raise EvidenceExtractionError("AI returned an empty response")

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise EvidenceExtractionError(
            "AI response was not valid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise EvidenceExtractionError(
            "AI response must be a JSON object"
        )

    required_fields = {
        "amount",
        "currency",
        "date",
        "status",
        "description",
        "fact_type",
        "confidence",
    }

    missing_fields = required_fields - data.keys()

    if missing_fields:
        raise EvidenceExtractionError(
            f"AI response is missing fields: {sorted(missing_fields)}"
        )

    amount = _validate_amount(data["amount"])
    currency = _validate_currency(data["currency"])
    date_value = _validate_date(data["date"])

    status = _clean_optional_string(data["status"])
    description = _clean_optional_string(data["description"])
    fact_type = _clean_optional_string(data["fact_type"])

    if fact_type is not None:
        fact_type = fact_type.lower()

        if fact_type not in ALLOWED_FACT_TYPES:
            raise EvidenceExtractionError(
                f"Unsupported extracted fact type: {fact_type}"
            )

    confidence = _validate_confidence(data["confidence"])

    return ExtractedEvidence(
        amount=amount,
        currency=currency,
        date=date_value,
        status=status,
        description=description,
        fact_type=fact_type,
        confidence=confidence,
    )


def extraction_to_evidence(
    extracted: ExtractedEvidence,
    evidence_id: str,
    user_id: str,
    request_id: str | None,
    event_id: str | None,
    source_type: str,
    content: str,
) -> Evidence:
    """
    Convert validated AI output into the project's Evidence model.
    """

    extracted_facts = {
        "amount": extracted.amount,
        "currency": extracted.currency,
        "date": extracted.date,
        "status": extracted.status,
        "description": extracted.description,
        "fact_type": extracted.fact_type,
    }

    return Evidence(
        evidence_id=evidence_id,
        user_id=user_id,
        request_id=request_id,
        event_id=event_id,
        source_type=source_type,
        content=content,
        extracted_facts=extracted_facts,
        confidence=extracted.confidence,
    )


class EvidenceExtractor:
    """
    High-level extraction service.

    This class deliberately does not make affordability decisions.
    """

    def __init__(self, provider: AIProvider):
        self.provider = provider

    def extract_image(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        evidence_id: str,
        user_id: str,
        request_id: str | None,
        event_id: str | None,
    ) -> Evidence:
        if not image_path.exists():
            raise EvidenceExtractionError(
                f"Evidence image does not exist: {image_path}"
            )

        raw_response = self.provider.extract_from_image(
            image_path=image_path,
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
        )

        extracted = parse_extraction_response(raw_response)

        return extraction_to_evidence(
            extracted=extracted,
            evidence_id=evidence_id,
            user_id=user_id,
            request_id=request_id,
            event_id=event_id,
            source_type="image",
            content=str(image_path),
        )

    def extract_message(
        self,
        message_text: str,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        evidence_id: str,
        user_id: str,
        request_id: str | None,
        event_id: str | None,
    ) -> Evidence:
        raw_response = self.provider.extract_from_text(
            text=message_text,
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
        )

        extracted = parse_extraction_response(raw_response)

        return extraction_to_evidence(
            extracted=extracted,
            evidence_id=evidence_id,
            user_id=user_id,
            request_id=request_id,
            event_id=event_id,
            source_type="message",
            content=message_text,
        )


class MockAIProvider:
    """
    Deterministic provider used only for local pipeline testing.

    It does not contain challenge answers.
    It simply returns a supplied JSON response so that the extraction
    and validation layers can be tested without an external AI service.
    """

    def __init__(
        self,
        image_response: str | None = None,
        text_response: str | None = None,
    ):
        self.image_response = image_response
        self.text_response = text_response

    def extract_from_image(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        if not image_path.exists():
            raise EvidenceExtractionError(
                f"Evidence image does not exist: {image_path}"
            )

        if self.image_response is None:
            raise EvidenceExtractionError(
                "MockAIProvider has no configured image response"
            )

        return self.image_response

    def extract_from_text(
        self,
        text: str,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        if self.text_response is None:
            raise EvidenceExtractionError(
                "MockAIProvider has no configured text response"
            )

        return self.text_response