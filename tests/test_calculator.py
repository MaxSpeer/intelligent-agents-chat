"""Tests for the safely-evaluated calculator tool."""

import unittest

from intelligent_agents_chat.tools.calculator import run


class CalculatorToolTests(unittest.TestCase):
    def test_basic_arithmetic_with_precedence_and_parentheses(self) -> None:
        self.assertEqual(run({"expression": "12 * (3 + 4)"}), "84")

    def test_division_returns_a_float(self) -> None:
        self.assertEqual(run({"expression": "7 / 2"}), "3.5")

    def test_supports_floor_division_modulo_and_power(self) -> None:
        self.assertEqual(run({"expression": "7 // 2"}), "3")
        self.assertEqual(run({"expression": "7 % 2"}), "1")
        self.assertEqual(run({"expression": "2 ** 10"}), "1024")

    def test_supports_unary_minus(self) -> None:
        self.assertEqual(run({"expression": "-5 + 3"}), "-2")

    def test_rounds_floating_point_noise(self) -> None:
        self.assertEqual(run({"expression": "0.1 + 0.2"}), "0.3")

    def test_division_by_zero_is_a_reported_error_not_an_exception(self) -> None:
        result = run({"expression": "1 / 0"})
        self.assertTrue(result.startswith("Error:"))

    def test_missing_expression_is_a_reported_error(self) -> None:
        result = run({})
        self.assertTrue(result.startswith("Error:"))

    def test_disallowed_constructs_are_rejected_not_executed(self) -> None:
        result = run({"expression": "__import__('os').system('echo pwned')"})
        self.assertTrue(result.startswith("Error:"))

    def test_string_and_name_operands_are_rejected(self) -> None:
        self.assertTrue(run({"expression": "'a' + 'b'"}).startswith("Error:"))
        self.assertTrue(run({"expression": "x + 1"}).startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
