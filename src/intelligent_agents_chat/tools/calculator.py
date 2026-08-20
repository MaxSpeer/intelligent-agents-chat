"""A basic arithmetic calculator tool, safely evaluated (no raw eval())."""

from __future__ import annotations

import ast
import operator

CALCULATOR_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": 'The arithmetic expression to evaluate, e.g. "12 * (3 + 4)".',
                },
            },
            "required": ["expression"],
        },
    },
}

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def run(arguments: dict) -> str:
    """Evaluate arguments["expression"] and return the result, or an error, as text."""
    expression = arguments.get("expression", "")
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except Exception as error:
        return f"Error: {error}"
    if isinstance(result, float):
        result = round(result, 10)
    return str(result)
