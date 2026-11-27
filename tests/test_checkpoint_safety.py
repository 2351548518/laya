"""Enforce the serialized artifact boundary for native checkpoint readers."""

import ast
from pathlib import Path


def check():
    package = Path(__file__).resolve().parents[1] / "laya_multimodal"
    invalid = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            module, operation = node.func.value, node.func.attr
            if not isinstance(module, ast.Name) or operation not in {"load", "loads"}:
                continue
            if module.id == "torch":
                if not any(key.arg == "weights_only" and isinstance(key.value, ast.Constant)
                           and key.value.value is True for key in node.keywords):
                    invalid.append(f"{path}:{node.lineno}")
            elif module.id in {"pickle", "joblib", "yaml"}:
                invalid.append(f"{path}:{node.lineno}")
    assert not invalid, invalid


if __name__ == "__main__":
    check()
