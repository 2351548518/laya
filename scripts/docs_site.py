"""Build or preview the English and Chinese documentation together."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CROSS_LANGUAGE_LINK = re.compile(r"(?<=\]\()(?:\.\./)+(?P<root>docs(?:_zh)?)/(?P<target>[^\s)]+)")


def stage_docs(destination: Path) -> None:
    staged = {"docs": destination, "docs_zh": destination / "zh-CN"}
    for name, directory in staged.items():
        shutil.copytree(ROOT / name, directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for page in destination.rglob("*.md"):
        source = page.read_text(encoding="utf-8")

        def site_link(match):
            target = staged[match["root"]] / match["target"]
            return os.path.relpath(target, page.parent).replace(os.sep, "/")

        updated = CROSS_LANGUAGE_LINK.sub(site_link, source)
        if updated != source:
            page.write_text(updated, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="Build the combined site with strict validation")
    preview = commands.add_parser("serve", help="Preview a fresh copy of both documentation trees")
    preview.add_argument("--dev-addr")
    preview.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    workspace = ROOT / "build"
    workspace.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="docs-", dir=workspace) as directory:
        stage = Path(directory) / "docs"
        stage_docs(stage)
        # Keep the temporary config at the repo root so Python API references and site/ resolve there.
        config = ROOT / f".zensical-{stage.parent.name}.toml"
        source = (ROOT / "zensical.toml").read_text(encoding="utf-8")
        source = source.replace("[project]\n", f'[project]\ndocs_dir = "{stage.relative_to(ROOT).as_posix()}"\n', 1)
        with config.open("x", encoding="utf-8") as stream:
            stream.write(source)
        command = [sys.executable, "-m", "zensical", args.command, "--config-file", str(config)]
        if args.command == "build":
            command.extend(["--strict", "--clean"])
        else:
            if args.dev_addr:
                command.extend(["--dev-addr", args.dev_addr])
            if args.open:
                command.append("--open")
        try:
            return subprocess.run(command, cwd=ROOT).returncode
        finally:
            config.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
