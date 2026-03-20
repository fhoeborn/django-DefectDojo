"""
LLM-based validation for risk acceptance statements.

When enabled via settings (LLM_RISK_ACCEPTANCE_VALIDATION_ENABLED), the
decision_details and recommendation_details fields are sent to an OpenAI-
compatible LLM to detect invalid or illogical text before a risk acceptance
is saved.  If the LLM is not configured or an error occurs the validation is
silently skipped so that the feature degrades gracefully.
"""
import json
import logging

import openai
from django.conf import settings
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a security risk management assistant. "
    "Your task is to evaluate whether a risk acceptance statement is valid "
    "and logically sound from a security perspective. "
    "A statement is invalid if it is: gibberish, empty of meaningful content, "
    "self-contradictory, logically incoherent, or clearly not related to "
    "security risk management. "
    "Respond with a JSON object containing exactly two keys: "
    '"valid" (boolean) and "reason" (string). '
    "If the statement is valid, set valid=true and provide a brief reason. "
    "If it is invalid, set valid=false and explain why."
)


def _get_openai_client():
    """Return an openai.OpenAI client configured from settings, or None."""
    api_key = getattr(settings, "LLM_OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("LLM_OPENAI_API_KEY is not configured; skipping LLM validation")
        return None
    return openai.OpenAI(api_key=api_key)


def _validate_statement(client, field_label: str, text: str) -> None:
    """Call the LLM for a single field and raise ValidationError when invalid.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client instance.
    field_label:
        Human-readable label used in error messages.
    text:
        The text to validate.

    Raises
    ------
    ValidationError
        When the LLM determines the statement is invalid or illogical.
    """
    model = getattr(settings, "LLM_OPENAI_MODEL", "gpt-4o-mini")
    user_message = (
        f"Please evaluate the following risk acceptance {field_label}:\n\n{text}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        if not result.get("valid", True):
            reason = result.get("reason", "The statement was deemed invalid by the LLM validator.")
            raise ValidationError(
                f"The {field_label} was flagged as invalid or illogical: {reason}"
            )
    except ValidationError:
        raise
    except Exception:
        logger.exception(
            "LLM validation encountered an error for field '%s'; skipping validation",
            field_label,
        )


def validate_risk_acceptance_statement(decision_details: str | None, recommendation_details: str | None) -> None:
    """Validate risk acceptance text fields using the configured LLM.

    This function is a no-op when:
    - ``LLM_RISK_ACCEPTANCE_VALIDATION_ENABLED`` is ``False`` (the default).
    - ``LLM_OPENAI_API_KEY`` is not set.
    - Any network or API error occurs (errors are logged but not re-raised).

    Parameters
    ----------
    decision_details:
        Contents of the ``decision_details`` model field.
    recommendation_details:
        Contents of the ``recommendation_details`` model field.

    Raises
    ------
    ValidationError
        When the LLM flags one or more fields as invalid or illogical.
    """
    if not getattr(settings, "LLM_RISK_ACCEPTANCE_VALIDATION_ENABLED", False):
        return

    client = _get_openai_client()
    if client is None:
        return

    errors = []
    for field_label, text in [
        ("decision details", decision_details),
        ("recommendation details", recommendation_details),
    ]:
        if text and text.strip():
            try:
                _validate_statement(client, field_label, text)
            except ValidationError as exc:
                errors.extend(exc.messages)

    if errors:
        raise ValidationError(errors)
