from utils.expense_calculator import Calculator
from typing import List
from langchain.tools import tool
import re


class CalculatorTool:
    def __init__(self):
        self.calculator = Calculator()
        self.calculator_tool_list = self._setup_tools()

    def _clean_number(self, value):
        """
        Converts:
        ₹2500 -> 2500
        2500 INR -> 2500
        3 days -> 3
        """
        if isinstance(value, (int, float)):
            return float(value)

        value = str(value)
        value = re.sub(r"[^\d.]", "", value)

        return float(value) if value else 0.0

    def _setup_tools(self) -> List:

        @tool
        def estimate_total_hotel_cost(
            price_per_night: str,
            total_days: float
        ) -> float:
            """Calculate total hotel cost"""

            price = self._clean_number(price_per_night)
            days = self._clean_number(total_days)

            return self.calculator.multiply(price, days)

        @tool
        def calculate_total_expense(*costs: float) -> float:
            """Calculate total expense of the trip"""

            cleaned_costs = [
                self._clean_number(cost)
                for cost in costs
            ]

            return self.calculator.calculate_total(*cleaned_costs)

        @tool
        def calculate_daily_expense_budget(
            total_cost: float,
            days: int
        ) -> float:
            """Calculate daily expense budget"""

            total_cost = self._clean_number(total_cost)
            days = int(self._clean_number(days))

            return self.calculator.calculate_daily_budget(
                total_cost,
                days
            )

        return [
            estimate_total_hotel_cost,
            calculate_total_expense,
            calculate_daily_expense_budget,
        ]