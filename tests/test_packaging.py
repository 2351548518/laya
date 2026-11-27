"""Check the shipped native command targets and local documentation links."""

import ast
from pathlib import Path
import re
import sys
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def headings(path):
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            slug = re.sub(r"[^\w\s-]", "", match.group(1).lower())
            result.add(re.sub(r"\s+", "-", slug))
        result.update(re.findall(r'<a (?:name|id)="([\w-]+)"></a>', line))
    return result


class PackagingTests(unittest.TestCase):
    def test_console_commands_resolve_to_shipped_functions(self):
        project = (ROOT / "pyproject.toml").read_text()
        paths = {
            "laya": "laya.cli", "laya-train": "laya.train_cli",
            "laya-mm-train": "laya.multimodal_train", "laya-serve": "laya.serve",
            "laya-evals": "laya.evals_cli", "laya-mcp-server": "laya.mcp.server",
        }
        for command, module in paths.items():
            with self.subTest(command=command):
                self.assertIn(f'{command} = "{module}:main"', project)
                tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text())
                self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body))

    def test_local_documentation_links_resolve(self):
        roots = ["docs", "examples", "benchmarks", "verify", "notebooks", "sdk"]
        documents = [ROOT / "README.md", ROOT / "LOCAL_SETUP.md", ROOT / "AGENTS.md", ROOT / "CONTRIBUTING.md"]
        for root in roots:
            documents.extend(path for path in (ROOT / root).rglob("*.md")
                             if all(part not in {"node_modules", "dist", "build"} for part in path.parts))
        missing = []
        anchors = {}
        for path in documents:
            for link in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)", path.read_text(encoding="utf-8")):
                if re.match(r"[a-z]+:", link) or link.startswith("/"):
                    continue
                name, _, fragment = unquote(link).partition("#")
                destination = path.parent / name if name else path
                if not destination.exists():
                    missing.append(f"{path.relative_to(ROOT)}: {link}")
                elif fragment and destination.suffix == ".md":
                    if destination not in anchors:
                        anchors[destination] = headings(destination)
                    if fragment not in anchors[destination]:
                        missing.append(f"{path.relative_to(ROOT)}: {link}")
        self.assertEqual(missing, [])

    def test_training_has_no_dependency_on_the_retired_encoder(self):
        from importlib.util import find_spec

        for module in ("laya.onnx_agent", "laya.fast", "laya.grounding_train"):
            self.assertIsNone(find_spec(module))


if __name__ == "__main__":
    unittest.main()
