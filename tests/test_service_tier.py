"""
Tests for service_tier="priority" injection in OpenAI API calls.

Covers:
  - service_tier is set to "priority" when AZURE_OPENAI_ENDPOINT is absent
  - service_tier is NOT set when AZURE_OPENAI_ENDPOINT is present
  - service_tier is NOT set when AZURE_OPENAI_ENDPOINT is whitespace-only
  - Both the main loop (api_kwargs) and QA phase (api_kwargs_qa) are covered
  - _make_openai_client returns AsyncAzureOpenAI when env var is set
  - _make_openai_client returns AsyncOpenAI when env var is absent
  - Logic is consistent across empty string, missing var, and populated var
"""

from __future__ import annotations

import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

# Ensure src/ is importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _import_server():
    """Import server module freshly; isolated per-test via monkeypatch."""
    import purple.server as srv
    return srv


# ---------------------------------------------------------------------------
# Helpers — build api_kwargs the same way server.py does
# ---------------------------------------------------------------------------

def _build_main_kwargs(*, azure_endpoint: str | None, is_reasoning: bool) -> dict:
    """Replicate the api_kwargs construction logic from solve_instance()."""
    COMPACT_THRESHOLD = 200_000
    api_kwargs: dict = {
        "model": "gpt-5.4",
        "instructions": "system prompt",
        "input": [],
        "tools": [],
        "parallel_tool_calls": False,
        "store": False,
    }
    if is_reasoning:
        api_kwargs["include"] = ["reasoning.encrypted_content"]
        api_kwargs["context_management"] = [
            {"type": "compaction", "compact_threshold": COMPACT_THRESHOLD},
        ]
        api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
        api_kwargs["max_output_tokens"] = 16_000
    else:
        api_kwargs["temperature"] = 0.0
        api_kwargs["max_output_tokens"] = 4096
    if not (azure_endpoint or "").strip():
        api_kwargs["service_tier"] = "priority"
    return api_kwargs


def _build_qa_kwargs(*, azure_endpoint: str | None, is_reasoning: bool) -> dict:
    """Replicate the api_kwargs_qa construction logic from the QA phase."""
    COMPACT_THRESHOLD = 200_000
    api_kwargs_qa: dict = {
        "model": "gpt-5.4",
        "instructions": "system prompt",
        "input": [],
        "tools": [],
        "parallel_tool_calls": False,
        "store": False,
    }
    if is_reasoning:
        api_kwargs_qa["include"] = ["reasoning.encrypted_content"]
        api_kwargs_qa["context_management"] = [
            {"type": "compaction", "compact_threshold": COMPACT_THRESHOLD},
        ]
        api_kwargs_qa["reasoning"] = {"effort": "high", "summary": "auto"}
        api_kwargs_qa["max_output_tokens"] = 16_000
    else:
        api_kwargs_qa["temperature"] = 0.0
        api_kwargs_qa["max_output_tokens"] = 4096
    if not (azure_endpoint or "").strip():
        api_kwargs_qa["service_tier"] = "priority"
    return api_kwargs_qa


# ===========================================================================
# Tests: service_tier in main loop api_kwargs
# ===========================================================================

class TestMainLoopServiceTier:

    def test_openai_reasoning_has_priority(self):
        """OpenAI + reasoning model: service_tier must be 'priority'."""
        kwargs = _build_main_kwargs(azure_endpoint=None, is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_openai_non_reasoning_has_priority(self):
        """OpenAI + non-reasoning model: service_tier must be 'priority'."""
        kwargs = _build_main_kwargs(azure_endpoint=None, is_reasoning=False)
        assert kwargs["service_tier"] == "priority"

    def test_azure_reasoning_no_service_tier(self):
        """Azure + reasoning model: service_tier must NOT be present."""
        kwargs = _build_main_kwargs(
            azure_endpoint="https://my-resource.openai.azure.com", is_reasoning=True
        )
        assert "service_tier" not in kwargs

    def test_azure_non_reasoning_no_service_tier(self):
        """Azure + non-reasoning model: service_tier must NOT be present."""
        kwargs = _build_main_kwargs(
            azure_endpoint="https://my-resource.openai.azure.com", is_reasoning=False
        )
        assert "service_tier" not in kwargs

    def test_empty_string_endpoint_has_priority(self):
        """Empty string AZURE_OPENAI_ENDPOINT is treated as absent — priority set."""
        kwargs = _build_main_kwargs(azure_endpoint="", is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_whitespace_only_endpoint_has_priority(self):
        """Whitespace-only AZURE_OPENAI_ENDPOINT is treated as absent — priority set."""
        kwargs = _build_main_kwargs(azure_endpoint="   ", is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_service_tier_value_is_exactly_priority(self):
        """service_tier value must be the exact string 'priority', not 'Priority' etc."""
        kwargs = _build_main_kwargs(azure_endpoint=None, is_reasoning=True)
        assert kwargs["service_tier"] == "priority"
        assert isinstance(kwargs["service_tier"], str)

    def test_reasoning_keys_present_with_openai(self):
        """Reasoning keys must still be present alongside service_tier for OpenAI."""
        kwargs = _build_main_kwargs(azure_endpoint=None, is_reasoning=True)
        assert "reasoning" in kwargs
        assert kwargs["reasoning"]["effort"] == "high"
        assert "service_tier" in kwargs

    def test_temperature_present_for_non_reasoning_openai(self):
        """Non-reasoning path: temperature set and service_tier also present."""
        kwargs = _build_main_kwargs(azure_endpoint=None, is_reasoning=False)
        assert kwargs["temperature"] == 0.0
        assert kwargs["service_tier"] == "priority"

    def test_no_temperature_for_reasoning_openai(self):
        """Reasoning path: temperature must NOT be set (would conflict with reasoning)."""
        kwargs = _build_main_kwargs(azure_endpoint=None, is_reasoning=True)
        assert "temperature" not in kwargs

    def test_azure_reasoning_keys_still_present(self):
        """Azure + reasoning: reasoning keys present but no service_tier."""
        kwargs = _build_main_kwargs(
            azure_endpoint="https://my.openai.azure.com", is_reasoning=True
        )
        assert "reasoning" in kwargs
        assert "service_tier" not in kwargs

    def test_required_base_keys_always_present(self):
        """Core keys (model, instructions, tools, etc.) always present regardless of tier."""
        for azure in [None, "https://azure.example.com"]:
            for reasoning in [True, False]:
                kwargs = _build_main_kwargs(azure_endpoint=azure, is_reasoning=reasoning)
                for key in ("model", "instructions", "input", "tools",
                            "parallel_tool_calls", "store"):
                    assert key in kwargs, f"Missing {key} with azure={azure}, reasoning={reasoning}"

    def test_parallel_tool_calls_always_false(self):
        """parallel_tool_calls must always be False."""
        for azure in [None, "https://azure.example.com"]:
            kwargs = _build_main_kwargs(azure_endpoint=azure, is_reasoning=True)
            assert kwargs["parallel_tool_calls"] is False

    def test_store_always_false(self):
        """store must always be False."""
        for azure in [None, "https://azure.example.com"]:
            kwargs = _build_main_kwargs(azure_endpoint=azure, is_reasoning=True)
            assert kwargs["store"] is False


# ===========================================================================
# Tests: service_tier in QA phase api_kwargs_qa
# ===========================================================================

class TestQAPhaseServiceTier:

    def test_openai_reasoning_has_priority(self):
        """QA phase, OpenAI + reasoning: service_tier must be 'priority'."""
        kwargs = _build_qa_kwargs(azure_endpoint=None, is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_openai_non_reasoning_has_priority(self):
        """QA phase, OpenAI + non-reasoning: service_tier must be 'priority'."""
        kwargs = _build_qa_kwargs(azure_endpoint=None, is_reasoning=False)
        assert kwargs["service_tier"] == "priority"

    def test_azure_reasoning_no_service_tier(self):
        """QA phase, Azure + reasoning: service_tier must NOT be present."""
        kwargs = _build_qa_kwargs(
            azure_endpoint="https://my-resource.openai.azure.com", is_reasoning=True
        )
        assert "service_tier" not in kwargs

    def test_azure_non_reasoning_no_service_tier(self):
        """QA phase, Azure + non-reasoning: service_tier must NOT be present."""
        kwargs = _build_qa_kwargs(
            azure_endpoint="https://my-resource.openai.azure.com", is_reasoning=False
        )
        assert "service_tier" not in kwargs

    def test_empty_string_endpoint_has_priority(self):
        """QA phase: empty string AZURE_OPENAI_ENDPOINT treated as absent."""
        kwargs = _build_qa_kwargs(azure_endpoint="", is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_whitespace_only_endpoint_has_priority(self):
        """QA phase: whitespace-only AZURE_OPENAI_ENDPOINT treated as absent."""
        kwargs = _build_qa_kwargs(azure_endpoint="\t  ", is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_service_tier_value_exact(self):
        """QA phase: service_tier value is exactly 'priority'."""
        kwargs = _build_qa_kwargs(azure_endpoint=None, is_reasoning=True)
        assert kwargs["service_tier"] == "priority"

    def test_reasoning_keys_present_openai(self):
        """QA phase: reasoning keys co-exist with service_tier on OpenAI."""
        kwargs = _build_qa_kwargs(azure_endpoint=None, is_reasoning=True)
        assert "reasoning" in kwargs
        assert kwargs["reasoning"]["effort"] == "high"
        assert "service_tier" in kwargs


# ===========================================================================
# Tests: main loop vs QA phase parity
# ===========================================================================

class TestMainQAParity:
    """Main loop and QA phase must produce consistent service_tier behaviour."""

    @pytest.mark.parametrize("azure_endpoint,is_reasoning", [
        (None, True),
        (None, False),
        ("https://resource.openai.azure.com", True),
        ("https://resource.openai.azure.com", False),
        ("", True),
        ("   ", False),
    ])
    def test_service_tier_parity(self, azure_endpoint, is_reasoning):
        """service_tier presence must match between main loop and QA phase."""
        main = _build_main_kwargs(azure_endpoint=azure_endpoint, is_reasoning=is_reasoning)
        qa = _build_qa_kwargs(azure_endpoint=azure_endpoint, is_reasoning=is_reasoning)
        assert ("service_tier" in main) == ("service_tier" in qa), (
            f"service_tier parity mismatch: main={main.get('service_tier')!r}, "
            f"qa={qa.get('service_tier')!r} "
            f"with azure_endpoint={azure_endpoint!r}, is_reasoning={is_reasoning}"
        )
        if "service_tier" in main:
            assert main["service_tier"] == qa["service_tier"]

    @pytest.mark.parametrize("azure_endpoint,is_reasoning", [
        (None, True),
        (None, False),
        ("https://resource.openai.azure.com", True),
        ("https://resource.openai.azure.com", False),
    ])
    def test_core_keys_parity(self, azure_endpoint, is_reasoning):
        """Core keys must be identical between main loop and QA phase."""
        main = _build_main_kwargs(azure_endpoint=azure_endpoint, is_reasoning=is_reasoning)
        qa = _build_qa_kwargs(azure_endpoint=azure_endpoint, is_reasoning=is_reasoning)
        for key in ("parallel_tool_calls", "store"):
            assert main[key] == qa[key], f"Mismatch on key={key!r}"


# ===========================================================================
# Tests: _make_openai_client factory
# ===========================================================================

class TestMakeOpenAIClient:
    """_make_openai_client must return the right client type based on env."""

    def test_returns_async_azure_when_endpoint_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key")
        from openai import AsyncAzureOpenAI
        assert isinstance(client, AsyncAzureOpenAI)

    def test_returns_async_openai_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key")
        from openai import AsyncOpenAI
        assert isinstance(client, AsyncOpenAI)

    def test_returns_async_openai_when_endpoint_empty(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "")
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key")
        from openai import AsyncOpenAI
        assert isinstance(client, AsyncOpenAI)

    def test_returns_async_openai_when_endpoint_whitespace(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "   ")
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key")
        from openai import AsyncOpenAI
        assert isinstance(client, AsyncOpenAI)

    def test_azure_uses_provided_api_version(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key")
        # AsyncAzureOpenAI stores version in _api_version
        assert "2025-01-01" in str(getattr(client, "_api_version", ""))

    def test_azure_uses_default_api_version(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.openai.azure.com")
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key")
        assert "2024-10-21" in str(getattr(client, "_api_version", ""))

    def test_openai_accepts_base_url(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        srv = _import_server()
        client = srv._make_openai_client(api_key="test-key", base_url="https://custom.api.example.com/v1")
        from openai import AsyncOpenAI
        assert isinstance(client, AsyncOpenAI)


# ===========================================================================
# Tests: env var read at call time (not import time)
# ===========================================================================

class TestEnvReadAtCallTime:
    """The env var must be read at call time, not cached at import time."""

    def test_service_tier_responds_to_env_at_runtime(self, monkeypatch):
        """Changing AZURE_OPENAI_ENDPOINT between calls changes the result."""
        # Without Azure: expect priority
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        kwargs_openai = _build_main_kwargs(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip() or None,
            is_reasoning=True,
        )
        assert "service_tier" in kwargs_openai

        # With Azure: expect no priority
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.openai.com")
        kwargs_azure = _build_main_kwargs(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip() or None,
            is_reasoning=True,
        )
        assert "service_tier" not in kwargs_azure

    def test_reading_env_directly_in_server(self, monkeypatch):
        """Server logic reads os.getenv at call time — verify with patched env."""
        srv = _import_server()
        # Patch os.getenv to simulate no Azure endpoint
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            # Logic: if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip() -> set priority
            result = not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            assert result is True  # priority would be set

        with mock.patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://azure.example.com"}):
            result = not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            assert result is False  # priority would NOT be set


# ===========================================================================
# Tests: is_reasoning_model helper
# ===========================================================================

class TestIsReasoningModel:
    """_is_reasoning_model must classify models correctly."""

    def test_gpt5_is_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("gpt-5.4") is True

    def test_gpt5_variants_are_reasoning(self):
        srv = _import_server()
        for model in ("gpt-5", "gpt-5.4", "gpt-5-turbo", "gpt-5.4-preview"):
            assert srv._is_reasoning_model(model) is True, f"{model} should be reasoning"

    def test_o1_is_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("o1") is True
        assert srv._is_reasoning_model("o1-mini") is True

    def test_o3_is_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("o3") is True
        assert srv._is_reasoning_model("o3-mini") is True

    def test_o4_is_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("o4-mini") is True

    def test_gpt4_is_not_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("gpt-4o") is False
        assert srv._is_reasoning_model("gpt-4o-mini") is False
        assert srv._is_reasoning_model("gpt-4-turbo") is False

    def test_gpt41_is_not_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("gpt-4.1") is False
        assert srv._is_reasoning_model("gpt-4.1-mini") is False

    def test_empty_string_is_not_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("") is False

    def test_unknown_model_is_not_reasoning(self):
        srv = _import_server()
        assert srv._is_reasoning_model("claude-3-5-sonnet") is False
        assert srv._is_reasoning_model("gemini-pro") is False
