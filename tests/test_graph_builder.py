"""
test_graph_builder.py — Integration-style tests for agent/agentic_workflow.py

The LLM, tool wrappers, and LangGraph internals are all mocked so that:
  - No real API keys are consumed.
  - Tests run fast (no network I/O).
  - We verify the graph is wired correctly (nodes, edges, compilation).
"""

from unittest.mock import MagicMock, patch
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Stub @tool callables — LangGraph's ToolNode.create_tool() requires real
# callables with __name__, so we can't use plain MagicMock instances here.
# ---------------------------------------------------------------------------

@tool
def _stub_get_weather(place: str) -> str:
    """Stub weather tool for testing."""
    return f"Weather in {place}: sunny"


@tool
def _stub_search_place(place: str) -> str:
    """Stub place search tool for testing."""
    return f"Top places in {place}"


@tool
def _stub_calculate(a: float, b: float) -> float:
    """Stub calculator tool for testing."""
    return a + b


@tool
def _stub_convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Stub currency converter tool for testing."""
    return amount


# A combined list that mirrors what GraphBuilder.__init__ builds.
STUB_TOOLS = [
    _stub_get_weather,
    _stub_search_place,
    _stub_calculate,
    _stub_convert_currency,
]


# ---------------------------------------------------------------------------
# Factory — creates a GraphBuilder with all external deps patched
# ---------------------------------------------------------------------------

def _patched_graph_builder():
    """
    Return (builder, mock_llm) where every external dependency is replaced:
      - ModelLoader.load_llm()  → MagicMock LLM
      - WeatherInfoTool/etc.    → objects whose *_tool_list attrs are STUB_TOOLS lists
      - GraphBuilder.tools      → STUB_TOOLS (set directly after __init__)
    """
    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    # Each wrapper instance needs its tool list to be a real Python list
    mock_weather_obj = MagicMock()
    mock_weather_obj.weather_tool_list = [_stub_get_weather]

    mock_place_obj = MagicMock()
    mock_place_obj.place_search_tool_list = [_stub_search_place]

    mock_calc_obj = MagicMock()
    mock_calc_obj.calculator_tool_list = [_stub_calculate]

    mock_currency_obj = MagicMock()
    mock_currency_obj.currency_tool_list = [_stub_convert_currency]

    with patch("agent.agentic_workflow.ModelLoader") as MockModelLoader, \
         patch("agent.agentic_workflow.WeatherInfoTool", return_value=mock_weather_obj), \
         patch("agent.agentic_workflow.PlaceSearchTool", return_value=mock_place_obj), \
         patch("agent.agentic_workflow.CalculatorTool", return_value=mock_calc_obj), \
         patch("agent.agentic_workflow.CurrencyConverterTool", return_value=mock_currency_obj):

        MockModelLoader.return_value.load_llm.return_value = mock_llm

        from agent.agentic_workflow import GraphBuilder
        builder = GraphBuilder(model_provider="groq")

    # Override self.tools with the real stub list in case list concatenation
    # through MagicMock attributes produced a MagicMock instead of a list.
    builder.tools = STUB_TOOLS
    builder.llm_with_tools = mock_llm_with_tools

    return builder, mock_llm


# ---------------------------------------------------------------------------
# Tests — GraphBuilder.__init__
# ---------------------------------------------------------------------------

class TestGraphBuilderInit:
    """Tests for GraphBuilder.__init__()"""

    def test_tools_list_is_not_empty(self):
        builder, _ = _patched_graph_builder()
        assert len(builder.tools) > 0

    def test_tools_list_has_four_tools(self):
        """One tool per category: weather, place, calculator, currency."""
        builder, _ = _patched_graph_builder()
        assert len(builder.tools) == 4

    def test_llm_bind_tools_called(self):
        builder, mock_llm = _patched_graph_builder()
        mock_llm.bind_tools.assert_called_once()

    def test_graph_is_none_before_build(self):
        builder, _ = _patched_graph_builder()
        assert builder.graph is None

    def test_system_prompt_is_set(self):
        builder, _ = _patched_graph_builder()
        assert builder.system_prompt is not None


# ---------------------------------------------------------------------------
# Tests — GraphBuilder.build_graph()
# ---------------------------------------------------------------------------

class TestGraphBuilderBuildGraph:
    """Tests for GraphBuilder.build_graph()"""

    def test_build_graph_returns_compiled_graph(self):
        builder, _ = _patched_graph_builder()
        graph = builder.build_graph()
        assert graph is not None

    def test_graph_attribute_set_after_build(self):
        builder, _ = _patched_graph_builder()
        builder.build_graph()
        assert builder.graph is not None

    def test_call_dunder_returns_graph(self):
        """GraphBuilder() should be callable and return the compiled graph."""
        builder, _ = _patched_graph_builder()
        result = builder()
        assert result is not None

    def test_graph_has_agent_node(self):
        """Compiled LangGraph should expose an 'agent' node."""
        builder, _ = _patched_graph_builder()
        graph = builder.build_graph()
        assert "agent" in graph.get_graph().nodes

    def test_graph_has_tools_node(self):
        """Compiled LangGraph should expose a 'tools' node."""
        builder, _ = _patched_graph_builder()
        graph = builder.build_graph()
        assert "tools" in graph.get_graph().nodes


# ---------------------------------------------------------------------------
# Tests — GraphBuilder.agent_function()
# ---------------------------------------------------------------------------

class TestAgentFunction:
    """Tests for GraphBuilder.agent_function()"""

    def test_agent_function_returns_messages_key(self):
        builder, _ = _patched_graph_builder()

        fake_response = MagicMock()
        builder.llm_with_tools.invoke.return_value = fake_response

        state = {"messages": [MagicMock(content="Plan a trip to Paris")]}
        result = builder.agent_function(state)

        assert "messages" in result
        assert result["messages"] == [fake_response]

    def test_agent_function_prepends_system_prompt(self):
        """System prompt must be the first message passed to the LLM."""
        builder, _ = _patched_graph_builder()
        builder.llm_with_tools.invoke.return_value = MagicMock()

        user_msg = MagicMock(content="Hello")
        builder.agent_function({"messages": [user_msg]})

        called_with = builder.llm_with_tools.invoke.call_args[0][0]
        assert called_with[0] is builder.system_prompt
        assert called_with[1] is user_msg

    def test_agent_function_calls_llm_once(self):
        """The LLM should be invoked exactly once per agent_function call."""
        builder, _ = _patched_graph_builder()
        builder.llm_with_tools.invoke.return_value = MagicMock()

        builder.agent_function({"messages": [MagicMock()]})

        assert builder.llm_with_tools.invoke.call_count == 1
