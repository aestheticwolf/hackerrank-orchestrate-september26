import json
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from code.ai.extractor import EvidenceExtractionError
from code.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiProvider:
    """Gemini implementation of the project's AIProvider interface."""

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

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

            contents: list[Any] = [
                types.Part.from_text(text=request_prompt)
            ]

            if image_path is not None:
                if not image_path.exists():
                    raise EvidenceExtractionError(
                        f"Image file does not exist: {image_path}"
                    )

                image_bytes = image_path.read_bytes()

                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    )
                )

            last_error: Exception | None = None

            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=schema,
                        ),
                    )

                    response_text = getattr(response, "text", None)

                    if not response_text:
                        raise EvidenceExtractionError(
                            "Gemini returned an empty response."
                        )

                    return response_text

                except EvidenceExtractionError:
                    raise

                except Exception as exc:
                    last_error = exc

                    if attempt == self.MAX_RETRIES:
                        break

                    time.sleep(
                        self.RETRY_DELAY_SECONDS * attempt
                    )

            raise EvidenceExtractionError(
                f"Gemini request failed after "
                f"{self.MAX_RETRIES} attempts: {last_error}"
            )

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