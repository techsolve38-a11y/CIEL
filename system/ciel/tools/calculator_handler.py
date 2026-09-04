"""
Calculator Tool Handler
---------------------------
A SAFE arithmetic evaluator — deliberately NOT the same thing as the
registered 'code_execution' tool, which implies running arbitrary code
and stays unimplemented on purpose until it can be done with real
sandboxing. This tool does one narrow thing safely: evaluate a math
expression like "23 * 47 + 12" and nothing else.

Why not just use Python's eval()? eval() would execute ANY Python code
in the string, including things like reading files, importing modules,
or worse — a classic, well-known security hole. Instead, this parses the
expression into an Abstract Syntax Tree and walks it manually, only
permitting numbers and basic arithmetic operators. Anything else
(function calls, attribute access, imports, variable names) is rejected
before it can run, not sanitized after the fact.
"""

from __future__ import annotations

import ast
import operator

# Only these operations are permitted — everything else is refused.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,   # unary minus, e.g. -5
    ast.Mod: operator.mod,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant not allowed: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_eval_node(node.operand))
    # Anything else — function calls, names, attributes, imports,
    # comprehensions, etc. — is explicitly refused, not silently ignored.
    raise ValueError(f"Expression contains a disallowed element: {type(node).__name__}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Cannot calculate '{expression}': division by zero."
    except (ValueError, SyntaxError, TypeError) as e:
        return f"Cannot calculate '{expression}': not a valid arithmetic expression ({e})."
