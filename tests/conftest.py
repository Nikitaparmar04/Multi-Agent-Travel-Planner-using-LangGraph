"""
conftest.py — Shared pytest fixtures for the Multi-Agent Trip Planner test suite.

Fixtures here are automatically available to all test files without importing.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Environment / API key fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """
    Inject fake API keys into the environment for every test so that no real
    API calls are made and no .env file is required during CI runs.
    """
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "fake-weather-key")
    monkeypatch.setenv("CURRENCY_API_KEY", "fake-currency-key")
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "fake-google-key")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")


# ---------------------------------------------------------------------------
# Reusable mock HTTP response factory
# ---------------------------------------------------------------------------

def make_mock_response(status_code: int = 200, json_data: dict = None):
    """Return a MagicMock that behaves like a requests.Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    return mock_resp
