import base64
import json
from pathlib import Path
from typing import Any

from google import genai

from code.ai.extractor import EvidenceExtractionError
from code.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiProvider:
    """Gemini implementation of the project's AIProvider interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key = api_key or GEMINI_API_KEY

        if not resolved_key:
            raise EvidenceExtractionError(
                "GEMINI_API_KEY is not configured."
            )

        self.model = model or GEMINI_MODEL
        self.client = genai.Client(api_key=resolved_key)

    def _generate(
        self,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        image_path: Path | None = None,
    ) -> str:
        """Send an evidence extraction request to Gemini."""

        try:
            request_prompt = (
                f"{prompt}\n\n"
                "Return ONLY valid JSON matching this schema:\n"
                f"{json.dumps(schema, separators=(',', ':'))}"
            )

            input_content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": request_prompt,
                }
            ]

            if image_path is not None:
                if not image_path.exists():
                    raise EvidenceExtractionError(
                        f"Image file does not exist: {image_path}"
                    )

                image_bytes = image_path.read_bytes()

                input_content.append(
                    {
                        "type": "image",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    }
                )

            response = self.client.interactions.create(
                model=self.model,
                input=input_content,
                system_instruction=system_prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
            )

            response_text = getattr(response, "output_text", None)

            if not response_text:
                raise EvidenceExtractionError(
                    "Gemini returned an empty response."
                )

            return response_text

        except EvidenceExtractionError:
            raise

        except Exception as exc:
            raise EvidenceExtractionError(
                f"Gemini evidence extraction failed: {exc}"
            ) from exc

    def extract_from_image(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        return self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            image_path=image_path,
        )

    def extract_from_text(
        self,
        text: str,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        combined_prompt = (
            f"{prompt}\n\n"
            "Evidence text:\n"
            f"{text}"
        )

        return self._generate(
            prompt=combined_prompt,
            system_prompt=system_prompt,
            schema=schema,
        )