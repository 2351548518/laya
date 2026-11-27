"""Check the shipped native command targets and local documentation links."""

import ast
from pathlib import Path
from posixpath import relpath
import re
import sys
import tomllib
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
    def test_documentation_language_switches_connect_each_pair(self):
        docs = ROOT / "docs"
        locale = docs / "zh-CN"
        english = {page.relative_to(docs) for page in docs.rglob("*.md") if not page.is_relative_to(locale)}
        chinese = {page.relative_to(locale) for page in locale.rglob("*.md")}
        self.assertEqual(english, chinese)
        for name in sorted(english):
            with self.subTest(page=name):
                en = docs / name
                zh = locale / name
                zh_link = relpath(zh.as_posix(), en.parent.as_posix())
                en_link = relpath(en.as_posix(), zh.parent.as_posix())
                self.assertIn(f"[简体中文]({zh_link})", en.read_text(encoding="utf-8"))
                self.assertIn(f"[English]({en_link})", zh.read_text(encoding="utf-8"))

    def test_console_commands_resolve_to_shipped_functions(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        project = config["project"]
        self.assertEqual(project["name"], "laya-multimodal")
        self.assertNotIn("laya", config["tool"]["setuptools"]["packages"])
        paths = {
            "laya-multimodal": "laya_multimodal.cli",
            "laya-multimodal-serve": "laya_multimodal.serve",
            "laya-multimodal-mcp-server": "laya_multimodal.mcp.server",
            "laya-multimodal-train-decisions": "laya_multimodal.multimodal_train",
        }
        self.assertEqual(project["scripts"], {cmd: f"{module}:main" for cmd, module in paths.items()})
        for command, module in paths.items():
            with self.subTest(command=command):
                tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text())
                self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body))

    def test_local_documentation_links_resolve(self):
        roots = ["docs", "examples", "benchmarks", "verify", "notebooks", "sdk"]
        documents = [ROOT / name for name in ("README.md", "README.zh-CN.md", "LOCAL_SETUP.md",
                                              "AGENTS.md", "CONTRIBUTING.md")]
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

        for module in ("laya_multimodal.onnx_agent", "laya_multimodal.fast", "laya_multimodal.grounding_train"):
            self.assertIsNone(find_spec(module))


if __name__ == "__main__":
    unittest.main()
