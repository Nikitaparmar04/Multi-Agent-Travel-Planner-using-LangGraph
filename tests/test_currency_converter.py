"""
test_currency_converter.py — Tests for utils/currency_converter.py

Uses unittest.mock to patch outbound HTTP requests so no real API calls
are made. Tests cover the happy path, API failure, and unknown currencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from utils.currency_converter import CurrencyConverter


FAKE_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "INR": 83.5,
    "GBP": 0.79,
}

FAKE_API_RESPONSE = {
    "conversion_rates": FAKE_RATES
}


@pytest.fixture
def converter():
    """Return a CurrencyConverter instance with a dummy API key."""
    return CurrencyConverter(api_key="fake-key-123")


class TestCurrencyConverterHappyPath:
    """Tests for successful currency conversion scenarios."""

    @patch("utils.currency_converter.requests.get")
    def test_convert_usd_to_inr(self, mock_get, converter):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_API_RESPONSE
        )
        result = converter.convert(100, "USD", "INR")
        assert result == pytest.approx(8350.0)

    @patch("utils.currency_converter.requests.get")
    def test_convert_usd_to_eur(self, mock_get, converter):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_API_RESPONSE
        )
        result = converter.convert(100, "USD", "EUR")
        assert result == pytest.approx(92.0)

    @patch("utils.currency_converter.requests.get")
    def test_convert_same_currency(self, mock_get, converter):
        """Converting USD→USD should return the same amount."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_API_RESPONSE
        )
        result = converter.convert(250, "USD", "USD")
        assert result == pytest.approx(250.0)

    @patch("utils.currency_converter.requests.get")
    def test_convert_fractional_amount(self, mock_get, converter):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_API_RESPONSE
        )
        result = converter.convert(0.5, "USD", "GBP")
        assert result == pytest.approx(0.395)

    @patch("utils.currency_converter.requests.get")
    def test_correct_url_is_called(self, mock_get, converter):
        """Verify that the API is called with the correct base URL + currency."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_API_RESPONSE
        )
        converter.convert(10, "EUR", "INR")
        called_url = mock_get.call_args[0][0]
        assert "EUR" in called_url


class TestCurrencyConverterErrors:
    """Tests for error / edge-case scenarios."""

    @patch("utils.currency_converter.requests.get")
    def test_api_failure_raises_exception(self, mock_get, converter):
        """Non-200 status code should raise an Exception."""
        mock_get.return_value = MagicMock(
            status_code=500,
            json=lambda: {"error": "server error"}
        )
        with pytest.raises(Exception):
            converter.convert(100, "USD", "INR")

    @patch("utils.currency_converter.requests.get")
    def test_unknown_target_currency_raises_value_error(self, mock_get, converter):
        """Requesting a currency not in rates should raise ValueError."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_API_RESPONSE
        )
        with pytest.raises(ValueError, match="XYZ not found"):
            converter.convert(100, "USD", "XYZ")
