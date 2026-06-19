import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from typing import List

load_dotenv()


class CurrencyConverterTool:
    def __init__(self):
        self.api_key = os.environ.get("EXCHANGE_RATE_API_KEY")
        self.currency_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the currency converter"""

        @tool
        def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
            """Convert an amount from one currency to another.

            Args:
                amount: The amount to convert.
                from_currency: The source currency code (e.g. USD, EUR, INR).
                to_currency: The target currency code (e.g. USD, EUR, INR).

            Returns:
                A string with the converted amount.
            """
            try:
                url = f"https://v6.exchangerate-api.com/v6/{self.api_key}/latest/{from_currency.upper()}"
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    return f"Currency conversion API error: {response.status_code}"
                data = response.json()
                rates = data.get("conversion_rates", {})
                to_upper = to_currency.upper()
                if to_upper not in rates:
                    return f"Currency '{to_currency}' not found in exchange rates."
                converted = amount * rates[to_upper]
                return (
                    f"{amount} {from_currency.upper()} = "
                    f"{converted:.2f} {to_upper} "
                    f"(Rate: 1 {from_currency.upper()} = {rates[to_upper]:.4f} {to_upper})"
                )
            except Exception as e:
                return f"Currency conversion failed: {str(e)}"

        return [convert_currency]
