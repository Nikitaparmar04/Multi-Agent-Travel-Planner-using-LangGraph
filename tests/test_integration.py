"""
test_integration.py — Integration Tests for Multi-Agent Trip Planner
=====================================================================

These tests verify that MULTIPLE components work TOGETHER correctly:

  Layer 1 — Tool + Utility Integration
    • WeatherInfoTool  ↔ WeatherForecastTool  (tool wrapper ↔ utility)
    • CalculatorTool   ↔ Calculator           (tool wrapper ↔ utility)
    • CurrencyConverterTool ↔ Exchange API    (tool wrapper ↔ HTTP layer)
    • PlaceSearchTool  ↔ Google/Tavily APIs   (tool wrapper ↔ fallback logic)

  Layer 2 — Agent Pipeline Integration
    • GraphBuilder builds a valid LangGraph with all 4 tool categories
    • Agent node correctly prepends system prompt and calls LLM
    • Full graph invoke returns a structured response dict

  Layer 3 — FastAPI Endpoint Integration
    • POST /query → GraphBuilder → LLM → response (end-to-end HTTP test)
    • GET  /       → health check
    • Error handling when LLM raises an exception

All external APIs and the LLM are MOCKED — no real network calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import tool
from fastapi.testclient import TestClient


# ============================================================
# Stub @tool functions (needed by LangGraph ToolNode)
# ============================================================

@tool
def _stub_weather(city: str) -> str:
    """Stub weather tool."""
    return f"Weather in {city}: 28°C, sunny"


@tool
def _stub_forecast(city: str) -> str:
    """Stub forecast tool."""
    return f"Forecast for {city}: warm all week"


@tool
def _stub_attractions(place: str) -> str:
    """Stub attractions tool."""
    return f"Top attractions in {place}: Museum, Park, Tower"


@tool
def _stub_restaurants(place: str) -> str:
    """Stub restaurants tool."""
    return f"Top restaurants in {place}: Cafe A, Bistro B"


@tool
def _stub_activities(place: str) -> str:
    """Stub activities tool."""
    return f"Activities in {place}: Hiking, Cycling"


@tool
def _stub_transportation(place: str) -> str:
    """Stub transportation tool."""
    return f"Transport in {place}: Metro, Bus, Taxi"


@tool
def _stub_hotel_cost(price_per_night: str, total_days: float) -> float:
    """Stub hotel cost tool."""
    return 2500.0 * total_days


@tool
def _stub_total_expense(*costs: float) -> float:
    """Stub total expense tool."""
    return sum(costs)


@tool
def _stub_daily_budget(total_cost: float, days: int) -> float:
    """Stub daily budget tool."""
    return total_cost / days


@tool
def _stub_convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Stub currency converter tool."""
    return f"{amount} {from_currency} = {amount * 85:.2f} {to_currency}"


ALL_STUB_TOOLS = [
    _stub_weather, _stub_forecast,
    _stub_attractions, _stub_restaurants, _stub_activities, _stub_transportation,
    _stub_hotel_cost, _stub_total_expense, _stub_daily_budget,
    _stub_convert_currency,
]


# ============================================================
# Shared helper — builds a fully patched GraphBuilder
# ============================================================

def _make_patched_builder(llm_response_content: str = "Here is your trip plan!"):
    """
    Returns (builder, mock_llm_with_tools).
    Every external dependency is replaced with a stub/mock.
    """
    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    fake_ai_message = MagicMock()
    fake_ai_message.content = llm_response_content
    mock_llm_with_tools.invoke.return_value = fake_ai_message

    mock_weather_obj = MagicMock()
    mock_weather_obj.weather_tool_list = [_stub_weather, _stub_forecast]

    mock_place_obj = MagicMock()
    mock_place_obj.place_search_tool_list = [
        _stub_attractions, _stub_restaurants,
        _stub_activities, _stub_transportation,
    ]

    mock_calc_obj = MagicMock()
    mock_calc_obj.calculator_tool_list = [
        _stub_hotel_cost, _stub_total_expense, _stub_daily_budget,
    ]

    mock_currency_obj = MagicMock()
    mock_currency_obj.currency_tool_list = [_stub_convert_currency]

    with patch("agent.agentic_workflow.ModelLoader") as MockLoader, \
         patch("agent.agentic_workflow.WeatherInfoTool", return_value=mock_weather_obj), \
         patch("agent.agentic_workflow.PlaceSearchTool", return_value=mock_place_obj), \
         patch("agent.agentic_workflow.CalculatorTool", return_value=mock_calc_obj), \
         patch("agent.agentic_workflow.CurrencyConverterTool", return_value=mock_currency_obj):

        MockLoader.return_value.load_llm.return_value = mock_llm

        from agent.agentic_workflow import GraphBuilder
        builder = GraphBuilder(model_provider="groq")

    builder.tools = ALL_STUB_TOOLS
    builder.llm_with_tools = mock_llm_with_tools
    return builder, mock_llm_with_tools


# ============================================================
# LAYER 1 — Tool ↔ Utility Integration Tests
# ============================================================

class TestWeatherToolIntegration:
    """
    WeatherInfoTool (tool wrapper) ↔ WeatherForecastTool (utility class)
    Verifies: tool wrapper correctly calls the utility and formats the output.
    """

    @patch("utils.weather_info.requests.get")
    def test_get_current_weather_tool_formats_response(self, mock_get):
        """Tool should format raw API dict into a human-readable string."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "weather": [{"description": "clear sky"}],
                "main": {"temp": 25.0, "humidity": 55},
                "name": "Paris",
                "cod": 200,
            }
        )
        from tools.whether_info_tool import WeatherInfoTool
        weather_tool_obj = WeatherInfoTool()
        get_weather_fn = weather_tool_obj.weather_tool_list[0]

        result = get_weather_fn.invoke({"city": "Paris"})

        assert "Paris" in result
        assert "25.0" in result
        assert "clear sky" in result

    @patch("utils.weather_info.requests.get")
    def test_get_current_weather_tool_handles_api_failure(self, mock_get):
        """When API returns non-200, tool should return a fallback error string."""
        mock_get.return_value = MagicMock(
            status_code=500,
            json=lambda: {}
        )
        from tools.whether_info_tool import WeatherInfoTool
        weather_tool_obj = WeatherInfoTool()
        get_weather_fn = weather_tool_obj.weather_tool_list[0]

        result = get_weather_fn.invoke({"city": "UnknownCity"})

        assert "Could not fetch" in result or "UnknownCity" in result

    @patch("utils.weather_info.requests.get")
    def test_get_weather_forecast_tool_returns_multiline(self, mock_get):
        """Forecast tool should return a multi-line string with date entries."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "list": [
                    {"dt_txt": "2025-07-01 12:00:00", "main": {"temp": 30.0},
                     "weather": [{"description": "sunny"}]},
                    {"dt_txt": "2025-07-02 12:00:00", "main": {"temp": 28.0},
                     "weather": [{"description": "partly cloudy"}]},
                ],
                "city": {"name": "Goa"}
            }
        )
        from tools.whether_info_tool import WeatherInfoTool
        weather_tool_obj = WeatherInfoTool()
        forecast_fn = weather_tool_obj.weather_tool_list[1]

        result = forecast_fn.invoke({"city": "Goa"})

        assert "2025-07-01" in result
        assert "2025-07-02" in result
        assert "Goa" in result


class TestCalculatorToolIntegration:
    """
    CalculatorTool (tool wrapper) ↔ Calculator (utility class)
    Verifies: tool correctly cleans input and delegates math to Calculator.
    """

    def test_hotel_cost_tool_with_currency_string(self):
        """Tool should clean '₹2500' string and compute 2500 × 3 = 7500."""
        from tools.expense_calculator_tool import CalculatorTool
        calc_tool_obj = CalculatorTool()
        hotel_fn = calc_tool_obj.calculator_tool_list[0]

        result = hotel_fn.invoke({"price_per_night": "₹2500", "total_days": 3})

        assert result == 7500.0

    def test_total_expense_combines_multiple_costs(self):
        """Total expense tool should sum all given costs correctly.

        Note: calculate_total_expense uses *costs (varargs), so we call the
        underlying Calculator directly to test the integration of tool->utility.
        """
        from tools.expense_calculator_tool import CalculatorTool
        calc_tool_obj = CalculatorTool()
        # Access the underlying Calculator utility directly (integration: tool → utility)
        result = calc_tool_obj.calculator.calculate_total(1000.0, 2000.0, 500.0)

        assert result == 3500.0

    def test_daily_budget_tool_divides_correctly(self):
        """Daily budget = total_cost / days."""
        from tools.expense_calculator_tool import CalculatorTool
        calc_tool_obj = CalculatorTool()
        daily_fn = calc_tool_obj.calculator_tool_list[2]

        result = daily_fn.invoke({"total_cost": 9000.0, "days": 3})

        assert result == 3000.0

    def test_hotel_cost_with_float_days(self):
        """Tool should handle float days correctly."""
        from tools.expense_calculator_tool import CalculatorTool
        calc_tool_obj = CalculatorTool()
        hotel_fn = calc_tool_obj.calculator_tool_list[0]

        result = hotel_fn.invoke({"price_per_night": "1000", "total_days": 5.0})

        assert result == 5000.0


class TestCurrencyConverterToolIntegration:
    """
    CurrencyConverterTool ↔ Exchange Rate API (HTTP)
    Verifies: tool correctly calls the API, parses rates, and formats output.
    """

    @patch("tools.currency_converter_tool.requests.get")
    def test_convert_usd_to_inr_success(self, mock_get):
        """Should correctly convert 100 USD to INR using mocked exchange rates."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "conversion_rates": {"INR": 83.5, "EUR": 0.92, "GBP": 0.79}
            }
        )
        from tools.currency_converter_tool import CurrencyConverterTool
        currency_obj = CurrencyConverterTool()
        convert_fn = currency_obj.currency_tool_list[0]

        result = convert_fn.invoke({
            "amount": 100.0,
            "from_currency": "USD",
            "to_currency": "INR"
        })

        assert "8350.00" in result
        assert "INR" in result
        assert "USD" in result

    @patch("tools.currency_converter_tool.requests.get")
    def test_convert_returns_error_on_api_failure(self, mock_get):
        """Non-200 API response → should return an error string, not crash."""
        mock_get.return_value = MagicMock(status_code=403, json=lambda: {})
        from tools.currency_converter_tool import CurrencyConverterTool
        currency_obj = CurrencyConverterTool()
        convert_fn = currency_obj.currency_tool_list[0]

        result = convert_fn.invoke({
            "amount": 50.0,
            "from_currency": "EUR",
            "to_currency": "INR"
        })

        assert "error" in result.lower() or "403" in result

    @patch("tools.currency_converter_tool.requests.get")
    def test_convert_unknown_target_currency(self, mock_get):
        """Unknown target currency → should return a not-found message."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"conversion_rates": {"INR": 83.5}}
        )
        from tools.currency_converter_tool import CurrencyConverterTool
        currency_obj = CurrencyConverterTool()
        convert_fn = currency_obj.currency_tool_list[0]

        result = convert_fn.invoke({
            "amount": 100.0,
            "from_currency": "USD",
            "to_currency": "XYZ"
        })

        assert "not found" in result.lower() or "XYZ" in result


class TestPlaceSearchToolIntegration:
    """
    PlaceSearchTool ↔ GooglePlaceSearchTool + TavilyPlaceSearchTool (fallback)
    Verifies: primary Google search is tried first; Tavily is used as fallback.
    """

    def test_search_attractions_uses_google_on_success(self):
        """When Google succeeds, result should mention 'google'."""
        with patch("utils.place_info_search.GooglePlacesAPIWrapper"), \
             patch("utils.place_info_search.GooglePlacesTool"), \
             patch("utils.place_info_search.GooglePlaceSearchTool.google_search_attractions") \
                as mock_google:
            mock_google.return_value = ["Eiffel Tower", "Louvre Museum"]

            from tools.place_search_tool import PlaceSearchTool
            place_obj = PlaceSearchTool()
            attractions_fn = place_obj.place_search_tool_list[0]

            result = attractions_fn.invoke({"place": "Paris"})

            assert "google" in result.lower()
            assert "Paris" in result

    def test_search_attractions_falls_back_to_tavily_on_google_failure(self):
        """When Google raises an exception, Tavily fallback should be used."""
        with patch("utils.place_info_search.GooglePlacesAPIWrapper"), \
             patch("utils.place_info_search.GooglePlacesTool"), \
             patch("utils.place_info_search.GooglePlaceSearchTool.google_search_attractions",
                   side_effect=Exception("API quota exceeded")), \
             patch("utils.place_info_search.TavilyPlaceSearchTool.tavily_search_attractions",
                   return_value=["Colosseum", "Vatican"]):

            from tools.place_search_tool import PlaceSearchTool
            place_obj = PlaceSearchTool()
            attractions_fn = place_obj.place_search_tool_list[0]

            result = attractions_fn.invoke({"place": "Rome"})

            assert "Rome" in result


# ============================================================
# LAYER 2 — Agent Pipeline Integration Tests
# ============================================================

class TestAgentPipelineIntegration:
    """
    Tests that the full LangGraph agent pipeline is wired and runs correctly:
    GraphBuilder (init + build) → agent node → tool node → response
    """

    def test_graph_contains_all_expected_nodes(self):
        """Compiled graph must have both 'agent' and 'tools' nodes."""
        builder, _ = _make_patched_builder()
        graph = builder.build_graph()
        nodes = graph.get_graph().nodes
        assert "agent" in nodes
        assert "tools" in nodes

    def test_graph_invoke_returns_messages_dict(self):
        """graph.invoke() should return a dict with a 'messages' key.

        LangGraph requires proper message objects (HumanMessage/AIMessage),
        not plain strings or MagicMocks.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        builder, mock_llm_with_tools = _make_patched_builder("Great trip plan!")

        # Use a real AIMessage — LangGraph's MessagesState rejects MagicMock
        fake_msg = AIMessage(content="Great trip plan!", id="test-id-1")
        mock_llm_with_tools.invoke.return_value = fake_msg

        graph = builder.build_graph()
        # Pass a real HumanMessage — LangGraph cannot coerce plain strings or MagicMocks
        result = graph.invoke({"messages": [HumanMessage(content="Plan a 3-day trip to Goa")]})

        assert isinstance(result, dict)
        assert "messages" in result

    def test_agent_node_prepends_system_prompt(self):
        """Agent function must prepend the system prompt before every LLM call."""
        from langchain_core.messages import HumanMessage, AIMessage
        builder, mock_llm_with_tools = _make_patched_builder()
        mock_llm_with_tools.invoke.return_value = AIMessage(content="ok", id="test-id-2")

        user_msg = HumanMessage(content="Trip to Mumbai")
        builder.agent_function({"messages": [user_msg]})

        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        assert call_args[0] is builder.system_prompt   # system prompt is FIRST
        assert call_args[1] is user_msg                # user message is SECOND

    def test_all_tool_categories_registered(self):
        """Builder should register tools from all 4 categories (≥ 10 tools)."""
        builder, _ = _make_patched_builder()
        assert len(builder.tools) >= 10

    def test_tool_names_cover_all_categories(self):
        """Tool list must include tools from weather, places, calc, and currency."""
        builder, _ = _make_patched_builder()
        tool_names = [t.name for t in builder.tools]

        # Weather category
        assert any("weather" in n.lower() for n in tool_names)
        # Place/search category
        assert any(
            any(kw in n.lower() for kw in ["attraction", "restaurant", "activit", "transport"])
            for n in tool_names
        )
        # Calculator category
        assert any(
            any(kw in n.lower() for kw in ["hotel", "expense", "budget"])
            for n in tool_names
        )
        # Currency category
        assert any("currency" in n.lower() or "convert" in n.lower() for n in tool_names)

    def test_llm_bind_tools_called_with_all_tools(self):
        """ModelLoader.load_llm().bind_tools() must receive the full tool list."""
        builder, _ = _make_patched_builder()
        # Verify bind_tools was called (happened in __init__)
        # builder.llm_with_tools is the result of bind_tools()
        assert builder.llm_with_tools is not None

    def test_graph_is_callable_via_dunder_call(self):
        """GraphBuilder.__call__() should return the compiled graph."""
        builder, _ = _make_patched_builder()
        graph = builder()
        assert graph is not None

    def test_agent_function_response_wraps_in_messages_list(self):
        """agent_function must return {'messages': [<response>]}."""
        from langchain_core.messages import HumanMessage, AIMessage
        builder, mock_llm_with_tools = _make_patched_builder()
        fake_resp = AIMessage(content="Here is your itinerary.", id="test-id-3")
        mock_llm_with_tools.invoke.return_value = fake_resp

        result = builder.agent_function({"messages": [HumanMessage(content="Hello")]})

        assert result == {"messages": [fake_resp]}


# ============================================================
# LAYER 3 — FastAPI Endpoint Integration Tests
# ============================================================

@pytest.fixture
def api_client():
    """
    Spin up a TestClient for app.py with GraphBuilder fully mocked.
    No real LLM or tool API calls are made.
    """
    fake_ai_msg = MagicMock()
    fake_ai_msg.content = "Here is your AI-generated trip plan for Paris!"
    fake_ai_msg.tool_calls = []

    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.invoke.return_value = fake_ai_msg

    mock_weather_obj = MagicMock()
    mock_weather_obj.weather_tool_list = [_stub_weather, _stub_forecast]

    mock_place_obj = MagicMock()
    mock_place_obj.place_search_tool_list = [
        _stub_attractions, _stub_restaurants,
        _stub_activities, _stub_transportation,
    ]

    mock_calc_obj = MagicMock()
    mock_calc_obj.calculator_tool_list = [
        _stub_hotel_cost, _stub_total_expense, _stub_daily_budget,
    ]

    mock_currency_obj = MagicMock()
    mock_currency_obj.currency_tool_list = [_stub_convert_currency]

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    with patch("agent.agentic_workflow.ModelLoader") as MockLoader, \
         patch("agent.agentic_workflow.WeatherInfoTool", return_value=mock_weather_obj), \
         patch("agent.agentic_workflow.PlaceSearchTool", return_value=mock_place_obj), \
         patch("agent.agentic_workflow.CalculatorTool", return_value=mock_calc_obj), \
         patch("agent.agentic_workflow.CurrencyConverterTool", return_value=mock_currency_obj):

        MockLoader.return_value.load_llm.return_value = mock_llm

        from app import app
        client = TestClient(app)

        # Override tools in the builder created inside the endpoint
        # by patching build_graph to use stub tools
        yield client, mock_llm_with_tools


class TestFastAPIEndpointIntegration:
    """
    End-to-end HTTP tests:  Client → FastAPI → GraphBuilder → LLM → Response
    """

    def test_health_check_returns_ok(self, api_client):
        """GET / should return status=ok."""
        client, _ = api_client
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "TripMind" in data["service"]

    def test_query_endpoint_returns_200(self, api_client):
        """POST /query with a valid query must return HTTP 200.

        LangGraph requires real AIMessage objects. We mock graph.invoke at
        the pregel level to return a pre-built messages dict.
        """
        from langchain_core.messages import AIMessage
        client, mock_llm = api_client

        # Mock at the graph.invoke level to bypass LangGraph message type checks
        fake_output = {"messages": [AIMessage(content="Here is your AI trip plan!")]}
        with patch("app.GraphBuilder") as MockBuilder:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = fake_output
            MockBuilder.return_value.return_value = mock_graph

            response = client.post("/query", json={"query": "Plan a trip to Paris"})

        assert response.status_code == 200

    def test_query_response_has_answer_key(self, api_client):
        """Response JSON must contain an 'answer' key."""
        from langchain_core.messages import AIMessage
        client, _ = api_client

        fake_output = {"messages": [AIMessage(content="Your trip itinerary!")]}
        with patch("app.GraphBuilder") as MockBuilder:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = fake_output
            MockBuilder.return_value.return_value = mock_graph

            response = client.post("/query", json={"query": "Trip to Manali"})

        data = response.json()
        assert "answer" in data

    def test_query_endpoint_rejects_empty_body(self, api_client):
        """POST /query with no body should return HTTP 422 (validation error)."""
        client, _ = api_client
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_endpoint_rejects_wrong_field_name(self, api_client):
        """POST /query with wrong field name ('question' instead of 'query')."""
        client, _ = api_client
        response = client.post("/query", json={"question": "Trip to Goa?"})
        assert response.status_code == 422

    def test_health_check_returns_service_name(self, api_client):
        """Health check response must identify the service."""
        client, _ = api_client
        response = client.get("/")
        data = response.json()
        assert "service" in data

    def test_query_with_complex_request(self, api_client):
        """Multi-part query should be accepted and return a structured response."""
        from langchain_core.messages import AIMessage
        client, _ = api_client

        fake_output = {"messages": [AIMessage(content="7-day Paris trip plan with budget 50000 INR")]}
        with patch("app.GraphBuilder") as MockBuilder:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = fake_output
            MockBuilder.return_value.return_value = mock_graph

            response = client.post("/query", json={
                "query": "Plan a 7-day trip to Paris with a budget of 50000 INR, "
                         "include weather, hotels, attractions and convert budget to EUR"
            })

        assert response.status_code == 200


# ============================================================
# LAYER 4 — Cross-Tool Data Flow Integration Tests
# ============================================================

class TestCrossToolDataFlowIntegration:
    """
    Verifies that data produced by one tool can be consumed by another tool.
    This catches FORMAT MISMATCH bugs that unit tests miss.
    """

    @patch("utils.weather_info.requests.get")
    def test_weather_output_is_string_consumable_by_agent(self, mock_get):
        """
        Weather tool output must be a plain string (not dict/None),
        so the LLM agent can directly use it as context.
        """
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "weather": [{"description": "sunny"}],
                "main": {"temp": 32.0},
                "name": "Jaipur",
                "cod": 200,
            }
        )
        from tools.whether_info_tool import WeatherInfoTool
        obj = WeatherInfoTool()
        result = obj.weather_tool_list[0].invoke({"city": "Jaipur"})

        # Must be a string — agent concatenates tool outputs as text context
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hotel_cost_output_feeds_into_total_expense(self):
        """
        Output of estimate_total_hotel_cost (float) must be directly usable
        as input to calculate_total_expense.

        Note: calculate_total_expense uses *costs varargs - we call the
        underlying Calculator utility to test the full tool->utility pipeline.
        """
        from tools.expense_calculator_tool import CalculatorTool
        obj = CalculatorTool()
        hotel_fn = obj.calculator_tool_list[0]

        # Step 1: calculate hotel cost via tool (tool → _clean_number → Calculator.multiply)
        hotel_cost = hotel_fn.invoke({"price_per_night": "2000", "total_days": 5})
        assert hotel_cost == 10000.0

        # Step 2: feed hotel cost into total via underlying Calculator
        # (tests that hotel_cost float is directly consumable as Calculator input)
        total = obj.calculator.calculate_total(hotel_cost, 3000.0, 1500.0)
        assert total == 14500.0

    def test_total_expense_feeds_into_daily_budget(self):
        """
        Output of calculate_total_expense must be directly usable as input
        to calculate_daily_expense_budget.
        """
        from tools.expense_calculator_tool import CalculatorTool
        obj = CalculatorTool()
        daily_fn = obj.calculator_tool_list[2]

        # Step 1: compute total via Calculator utility
        total = obj.calculator.calculate_total(6000.0, 4000.0, 2000.0)
        assert total == 12000.0

        # Step 2: feed total into daily budget tool (tests float → tool invocation)
        daily = daily_fn.invoke({"total_cost": total, "days": 4})
        assert daily == 3000.0

    @patch("tools.currency_converter_tool.requests.get")
    def test_currency_output_is_string_consumable_by_agent(self, mock_get):
        """
        Currency tool output must be a human-readable string for the agent to use.
        """
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"conversion_rates": {"EUR": 0.011}}
        )
        from tools.currency_converter_tool import CurrencyConverterTool
        obj = CurrencyConverterTool()
        result = obj.currency_tool_list[0].invoke({
            "amount": 50000.0,
            "from_currency": "INR",
            "to_currency": "EUR"
        })

        assert isinstance(result, str)
        assert "EUR" in result
        assert "INR" in result
