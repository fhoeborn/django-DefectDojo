"""
Unit tests for the LLM-based risk acceptance validation.

These tests exercise ``dojo.risk_acceptance.llm_validator`` in isolation using
``unittest.mock`` so that no real network calls are made.
"""
import json
from unittest.mock import MagicMock, patch

from django.core.exceptions import ValidationError
from django.test import TestCase, override_settings

from dojo.risk_acceptance.llm_validator import (
    _validate_statement,
    validate_risk_acceptance_statement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(valid: bool, reason: str = "test reason") -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    content = json.dumps({"valid": valid, "reason": reason})
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# validate_risk_acceptance_statement – feature disabled (default)
# ---------------------------------------------------------------------------

class TestLLMValidationDisabled(TestCase):
    """When the feature flag is off no LLM calls should be made."""

    @override_settings(LLM_RISK_ACCEPTANCE_VALIDATION_ENABLED=False)
    def test_disabled_does_not_call_llm(self):
        with patch("dojo.risk_acceptance.llm_validator._get_openai_client") as mock_client:
            # Should return without calling the client at all
            validate_risk_acceptance_statement("gibberish @@@ ###", "more gibberish")
        mock_client.assert_not_called()

    @override_settings(LLM_RISK_ACCEPTANCE_VALIDATION_ENABLED=False)
    def test_disabled_does_not_raise(self):
        # Must not raise even for completely nonsensical strings
        validate_risk_acceptance_statement("!!!", "???")


# ---------------------------------------------------------------------------
# validate_risk_acceptance_statement – feature enabled
# ---------------------------------------------------------------------------

@override_settings(
    LLM_RISK_ACCEPTANCE_VALIDATION_ENABLED=True,
    LLM_OPENAI_API_KEY="sk-test",
    LLM_OPENAI_MODEL="gpt-4o-mini",
)
class TestLLMValidationEnabled(TestCase):

    def _mock_client(self, decision_valid: bool = True, recommendation_valid: bool = True):
        """Return a patched openai client that answers sequentially."""
        client = MagicMock()
        responses = []
        if decision_valid is not None:
            responses.append(_make_openai_response(decision_valid, "decision reason"))
        if recommendation_valid is not None:
            responses.append(_make_openai_response(recommendation_valid, "recommendation reason"))
        client.chat.completions.create.side_effect = responses
        return client

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_valid_statements_do_not_raise(self, mock_get_client):
        client = self._mock_client(decision_valid=True, recommendation_valid=True)
        mock_get_client.return_value = client
        # Must not raise
        validate_risk_acceptance_statement(
            "We accept this risk because compensating controls are in place.",
            "The finding should be fixed within the next release cycle.",
        )
        self.assertEqual(client.chat.completions.create.call_count, 2)

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_invalid_decision_details_raises(self, mock_get_client):
        client = self._mock_client(decision_valid=False, recommendation_valid=True)
        mock_get_client.return_value = client
        with self.assertRaises(ValidationError) as cm:
            validate_risk_acceptance_statement(
                "asdfgh qwerty 12345",
                "The finding should be fixed in next sprint.",
            )
        self.assertIn("decision details", str(cm.exception))

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_invalid_recommendation_details_raises(self, mock_get_client):
        client = self._mock_client(decision_valid=True, recommendation_valid=False)
        mock_get_client.return_value = client
        with self.assertRaises(ValidationError) as cm:
            validate_risk_acceptance_statement(
                "We accept this risk because mitigations are in place.",
                "xyz abc 999 !!!",
            )
        self.assertIn("recommendation details", str(cm.exception))

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_both_invalid_raises_with_two_messages(self, mock_get_client):
        client = self._mock_client(decision_valid=False, recommendation_valid=False)
        mock_get_client.return_value = client
        with self.assertRaises(ValidationError) as cm:
            validate_risk_acceptance_statement("bad decision", "bad recommendation")
        messages = cm.exception.messages
        self.assertEqual(len(messages), 2)

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_none_fields_are_skipped(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        # None fields must be skipped without calling the LLM
        validate_risk_acceptance_statement(None, None)
        client.chat.completions.create.assert_not_called()

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_empty_string_fields_are_skipped(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        validate_risk_acceptance_statement("", "   ")
        client.chat.completions.create.assert_not_called()

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_llm_api_error_is_silenced(self, mock_get_client):
        """Any exception from the LLM call must be caught and not re-raised."""
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network error")
        mock_get_client.return_value = client
        # Should not raise
        validate_risk_acceptance_statement(
            "We accept this risk.",
            "Fix it next quarter.",
        )

    @patch("dojo.risk_acceptance.llm_validator._get_openai_client")
    def test_no_api_key_skips_validation(self, mock_get_client):
        mock_get_client.return_value = None
        # Must not raise
        validate_risk_acceptance_statement("gibberish", "more gibberish")


# ---------------------------------------------------------------------------
# _validate_statement (unit tests for the low-level helper)
# ---------------------------------------------------------------------------

class TestValidateStatement(TestCase):

    def _client(self, valid: bool, reason: str = "reason") -> MagicMock:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(valid, reason)
        return client

    @override_settings(LLM_OPENAI_MODEL="gpt-4o-mini")
    def test_valid_statement_does_not_raise(self):
        _validate_statement(
            self._client(True),
            "decision details",
            "We accept this risk because compensating controls are in place.",
        )

    @override_settings(LLM_OPENAI_MODEL="gpt-4o-mini")
    def test_invalid_statement_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            _validate_statement(
                self._client(False, "The statement is gibberish."),
                "decision details",
                "qwerty asdf 1234",
            )

    @override_settings(LLM_OPENAI_MODEL="gpt-4o-mini")
    def test_error_message_contains_field_label_and_reason(self):
        reason = "The statement is illogical."
        with self.assertRaises(ValidationError) as cm:
            _validate_statement(
                self._client(False, reason),
                "recommendation details",
                "xyz",
            )
        message = str(cm.exception)
        self.assertIn("recommendation details", message)
        self.assertIn(reason, message)
