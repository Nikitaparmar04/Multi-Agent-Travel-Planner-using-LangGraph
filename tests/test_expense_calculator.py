"""
test_expense_calculator.py — Unit tests for utils/expense_calculator.py

These are pure unit tests: no network calls, no mocks needed.
The Calculator class only does arithmetic, so tests are fast and deterministic.
"""

import pytest
from utils.expense_calculator import Calculator


class TestCalculatorMultiply:
    """Tests for Calculator.multiply()"""

    def test_multiply_positive_integers(self):
        assert Calculator.multiply(3, 4) == 12.0

    def test_multiply_returns_float(self):
        result = Calculator.multiply(2, 5)
        assert isinstance(result, float)

    def test_multiply_with_float_inputs(self):
        assert Calculator.multiply(2.5, 4.0) == pytest.approx(10.0)

    def test_multiply_by_zero(self):
        assert Calculator.multiply(100, 0) == 0.0

    def test_multiply_negative_numbers(self):
        assert Calculator.multiply(-3, 5) == -15.0

    def test_multiply_string_numbers(self):
        """Should coerce numeric strings to float."""
        assert Calculator.multiply("3", "4") == 12.0


class TestCalculatorTotal:
    """Tests for Calculator.calculate_total()"""

    def test_total_single_value(self):
        assert Calculator.calculate_total(42) == 42.0

    def test_total_multiple_values(self):
        assert Calculator.calculate_total(100, 200, 300) == 600.0

    def test_total_with_floats(self):
        assert Calculator.calculate_total(10.5, 20.5) == pytest.approx(31.0)

    def test_total_with_zeros(self):
        assert Calculator.calculate_total(0, 0, 0) == 0.0

    def test_total_string_numbers(self):
        """Should coerce numeric strings to float."""
        assert Calculator.calculate_total("50", "150") == 200.0

    def test_total_no_args(self):
        """Sum of nothing should be 0."""
        assert Calculator.calculate_total() == 0.0


class TestCalculatorDailyBudget:
    """Tests for Calculator.calculate_daily_budget()"""

    def test_daily_budget_basic(self):
        assert Calculator.calculate_daily_budget(1000, 5) == pytest.approx(200.0)

    def test_daily_budget_returns_float(self):
        result = Calculator.calculate_daily_budget(500, 4)
        assert isinstance(result, float)

    def test_daily_budget_single_day(self):
        assert Calculator.calculate_daily_budget(300, 1) == pytest.approx(300.0)

    def test_daily_budget_zero_days_returns_zero(self):
        """Dividing by zero days should safely return 0 (not raise)."""
        assert Calculator.calculate_daily_budget(500, 0) == 0.0

    def test_daily_budget_fractional_total(self):
        assert Calculator.calculate_daily_budget(100.5, 3) == pytest.approx(33.5)

    def test_daily_budget_string_inputs(self):
        """Should handle string-encoded numbers."""
        assert Calculator.calculate_daily_budget("900", "3") == pytest.approx(300.0)

    def test_daily_budget_float_days_string(self):
        """Float-encoded day strings like '5.0' should be accepted."""
        assert Calculator.calculate_daily_budget(1000, "5.0") == pytest.approx(200.0)
