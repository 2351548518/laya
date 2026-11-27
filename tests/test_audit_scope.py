"""The dependency audit's blocking scope matches the native container."""

from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AuditScopeTests(unittest.TestCase):
    @unittest.skipUnless(sys.version_info >= (3, 11), "the audit job uses Python 3.11+")
    def test_audit_reads_the_dependencies_installed_in_docker(self):
        workflow = (ROOT / ".github/workflows/security.yml").read_text()
        docker = (ROOT / "Dockerfile").read_text()
        extras = re.search(r'pip install "\.\[([^\]]+)\]"', docker)
        self.assertIsNotNone(extras)
        import tomllib
        table = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        installed = set(table["dependencies"])
        for name in extras.group(1).split(","):
            installed.update(table["optional-dependencies"][name])
        emitted = {}
        for name, ending in (("core", "PY"), ("extras", "PY")):
            expression = rf"<<'{ending}'\s*>\s*requirements-audit-{name}\.txt\n(.*?)(?:\n +{ending})"
            source = re.search(expression, workflow, re.S)
            self.assertIsNotNone(source, name)
            from textwrap import dedent
            result = subprocess.run([sys.executable, "-c", dedent(source.group(1))], text=True,
                                    cwd=ROOT, capture_output=True, check=True)
            emitted[name] = set(result.stdout.splitlines())
        all_specs = set(table["dependencies"])
        for requirements in table["optional-dependencies"].values():
            all_specs.update(requirements)
        self.assertEqual(emitted["core"], installed)
        self.assertEqual(emitted["extras"], all_specs - installed)


if __name__ == "__main__":
    unittest.main()
