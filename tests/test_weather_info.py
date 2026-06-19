"""
test_weather_info.py — Tests for utils/weather_info.py

All outbound HTTP requests are patched so no real OpenWeatherMap API key
is needed. Tests cover both current-weather and forecast endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from utils.weather_info import WeatherForecastTool


FAKE_CURRENT_WEATHER = {
    "weather": [{"description": "clear sky"}],
    "main": {"temp": 298.15, "humidity": 60},
    "name": "Paris",
    "cod": 200,
}

FAKE_FORECAST = {
    "list": [
        {"dt_txt": "2025-06-20 12:00:00", "main": {"temp": 22.0}},
        {"dt_txt": "2025-06-20 15:00:00", "main": {"temp": 25.0}},
    ],
    "city": {"name": "Paris"},
}


@pytest.fixture
def weather_tool():
    """Return a WeatherForecastTool with a dummy API key."""
    return WeatherForecastTool(api_key="fake-weather-key")


class TestGetCurrentWeather:
    """Tests for WeatherForecastTool.get_current_weather()"""

    @patch("utils.weather_info.requests.get")
    def test_returns_weather_dict_on_success(self, mock_get, weather_tool):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_CURRENT_WEATHER
        )
        result = weather_tool.get_current_weather("Paris")
        assert isinstance(result, dict)
        assert result["name"] == "Paris"

    @patch("utils.weather_info.requests.get")
    def test_returns_empty_dict_on_api_error(self, mock_get, weather_tool):
        """Non-200 response should return an empty dict, not raise."""
        mock_get.return_value = MagicMock(
            status_code=404,
            json=lambda: {"message": "city not found"}
        )
        result = weather_tool.get_current_weather("InvalidCity")
        assert result == {}

    @patch("utils.weather_info.requests.get")
    def test_correct_endpoint_called(self, mock_get, weather_tool):
        """Should hit the /weather endpoint."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_CURRENT_WEATHER
        )
        weather_tool.get_current_weather("London")
        called_url = mock_get.call_args[0][0]
        assert "/weather" in called_url

    @patch("utils.weather_info.requests.get")
    def test_api_key_passed_in_params(self, mock_get, weather_tool):
        """The API key should be included in the request params."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_CURRENT_WEATHER
        )
        weather_tool.get_current_weather("Tokyo")
        call_kwargs = mock_get.call_args[1]  # keyword args
        assert call_kwargs["params"]["appid"] == "fake-weather-key"

    @patch("utils.weather_info.requests.get")
    def test_network_exception_is_reraised(self, mock_get, weather_tool):
        """Exceptions from requests should bubble up."""
        mock_get.side_effect = ConnectionError("Network unreachable")
        with pytest.raises(ConnectionError):
            weather_tool.get_current_weather("Paris")


class TestGetForecastWeather:
    """Tests for WeatherForecastTool.get_forecast_weather()"""

    @patch("utils.weather_info.requests.get")
    def test_returns_forecast_dict_on_success(self, mock_get, weather_tool):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_FORECAST
        )
        result = weather_tool.get_forecast_weather("Paris")
        assert isinstance(result, dict)
        assert "list" in result
        assert len(result["list"]) == 2

    @patch("utils.weather_info.requests.get")
    def test_returns_empty_dict_on_api_error(self, mock_get, weather_tool):
        mock_get.return_value = MagicMock(
            status_code=401,
            json=lambda: {"message": "unauthorized"}
        )
        result = weather_tool.get_forecast_weather("Paris")
        assert result == {}

    @patch("utils.weather_info.requests.get")
    def test_correct_endpoint_called(self, mock_get, weather_tool):
        """Should hit the /forecast endpoint."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_FORECAST
        )
        weather_tool.get_forecast_weather("Berlin")
        called_url = mock_get.call_args[0][0]
        assert "/forecast" in called_url

    @patch("utils.weather_info.requests.get")
    def test_metric_units_requested(self, mock_get, weather_tool):
        """Forecast should request metric units."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: FAKE_FORECAST
        )
        weather_tool.get_forecast_weather("Delhi")
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["params"]["units"] == "metric"
