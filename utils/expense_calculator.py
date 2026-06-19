class Calculator:

    @staticmethod
    def multiply(a, b) -> float:
        """
        Multiply two numbers safely.
        """
        return float(a) * float(b)

    @staticmethod
    def calculate_total(*x) -> float:
        """
        Calculate sum of given numbers.
        """
        return sum(float(i) for i in x)

    @staticmethod
    def calculate_daily_budget(total, days) -> float:
        """
        Calculate daily budget.
        """
        total = float(total)
        days = int(float(days))

        return total / days if days > 0 else 0
