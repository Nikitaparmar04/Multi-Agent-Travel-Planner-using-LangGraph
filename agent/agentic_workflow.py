from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from tools.whether_info_tool import WeatherInfoTool          # actual filename
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_converter_tool import CurrencyConverterTool


class GraphBuilder:

    def __init__(self, model_provider: str = "groq"):

        self.model_loader = ModelLoader(
            model_provider=model_provider
        )

        self.llm = self.model_loader.load_llm()

        # Initialize tool-wrapper instances
        self.weather_tool_obj = WeatherInfoTool()
        self.place_search_tool_obj = PlaceSearchTool()
        self.calculator_tool_obj = CalculatorTool()
        self.currency_converter_tool_obj = CurrencyConverterTool()

        # Flatten all tool lists into one list
        # Each wrapper class exposes a *_tool_list attribute that is a list of @tool functions
        self.tools = (
            self.weather_tool_obj.weather_tool_list
            + self.place_search_tool_obj.place_search_tool_list
            + self.calculator_tool_obj.calculator_tool_list
            + self.currency_converter_tool_obj.currency_tool_list
        )

        # Bind tools with LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_prompt = SYSTEM_PROMPT
        self.graph = None

    def agent_function(self, state: MessagesState):
        """
        Main agent function
        """
        user_messages = state["messages"]
        input_messages = [self.system_prompt] + user_messages
        response = self.llm_with_tools.invoke(input_messages)
        return {"messages": [response]}

    def build_graph(self):

        graph_builder = StateGraph(MessagesState)

        # Add nodes
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(self.tools))

        # Define graph flow
        graph_builder.add_edge(START, "agent")

        graph_builder.add_conditional_edges(
            "agent",
            tools_condition
        )

        graph_builder.add_edge("tools", "agent")

        self.graph = graph_builder.compile()
        return self.graph

    def __call__(self):
        return self.build_graph()
