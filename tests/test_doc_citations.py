"""Check Python symbol citations in the repository's source and Markdown.

A `file.py::Owner.symbol` citation must resolve to that owner and symbol. A line
range paired with a symbol must lie inside its definition. Positive controls use
the current AST so the same checks cover both forms after code moves.

Run without weights or network using `python tests/test_doc_citations.py`.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")

PASS: List[str] = []
FAIL: List[str] = []


def check_true(what: str, cond: bool, detail: object = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def _shown(named: Sequence["Symbol"]) -> str:
    return ", ".join(str(s) for s in named)


class Symbol:
    """A name a ``::`` citation could mean, and the scope that defines it."""

    __slots__ = ("name", "parent", "start", "end")

    def __init__(self, name: str, parent: Optional[str], start: int, end: int) -> None:
        self.name, self.parent, self.start, self.end = name, parent, start, end

    def __str__(self) -> str:
        return "%s%s (%d-%d)" % ("%s." % self.parent if self.parent else "",
                                 self.name, self.start, self.end)


DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _collect(stmts: Sequence[ast.stmt], parent: Optional[str], out: List[Symbol]) -> None:
    for node in stmts:
        if isinstance(node, DEFS):
            out.append(Symbol(node.name, parent, node.lineno, node.end_lineno or node.lineno))
            _collect(node.body, node.name, out)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(target, ast.Name):
                    out.append(Symbol(target.id, parent, node.lineno,
                                      node.end_lineno or node.lineno))
        elif isinstance(node, ast.If):
            # A module-level guard (``if TYPE_CHECKING:``) still defines what it holds.
            _collect(node.body, parent, out)
            _collect(node.orelse, parent, out)


_CACHE: Dict[str, List[Symbol]] = {}


def symbols(rel: str) -> List[Symbol]:
    if rel not in _CACHE:
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        try:
            out: List[Symbol] = []
            _collect(ast.parse(open(path, encoding="utf-8").read()).body, None, out)
        except (OSError, SyntaxError, UnicodeDecodeError):
            out = []
        _CACHE[rel] = out
    return _CACHE[rel]


DBLCOLON = re.compile(r'(?P<file>[A-Za-z0-9_.\-/]+\.py)::(?P<name>[A-Za-z_][A-Za-z0-9_]*'
                      r'(?:\.[A-Za-z_][A-Za-z0-9_]*)?)')
LINECITE = re.compile(r'(?P<file>[A-Za-z0-9_.\-/]+\.py):(?P<a>\d+)(?:\s*-\s*(?P<b>\d+))?')
IDENT = re.compile(r'`(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`')
# `docs/index.md`'s deferral sentence: "<topics> is/are in the [README](...) for now."
DEFERRAL = re.compile(r'(?P<topics>[A-Za-z0-9 ,&/]+?)\s+(?:is|are) in the\s+'
                      r'\[README\]\([^)]*\)\s*for now\.')

# What may sit between a backticked symbol and the citation after it. Punctuation and spacing
# cover both live idioms -- "`Foo.bar` (`file:12`)" and "`Foo.bar`, (`file:12`)" -- and the
# connectors cover "`Router.predict_batch` -- see laya_multimodal/decisions/routing.py:961". Longer than that is a new
# clause, and binding across a clause is how a checker starts inventing violations.
GAP_LIMIT = 24
CONNECTORS = {"see", "via", "at", "in", "as", "is", "where", "shows", "says", "and"}

SKIP_DIRS = {".git", ".venv", ".venv-mm", "node_modules", "__pycache__", "dist", "build",
             ".mypy_cache", ".pytest_cache", ".cache", "site-packages", "site",
             "_local", "multimodal-data", "multimodal-runs"}


def target_of(cited: str) -> Optional[str]:
    """Resolve a cited path in the tree, tolerating the ``laya_multimodal/x.py`` and ``x.py`` spellings."""
    for cand in ({cited, cited.split("/", 1)[-1]} if "/" in cited else {cited}):
        if cand.endswith(".py") and os.path.isfile(os.path.join(ROOT, cand.replace("/", os.sep))):
            return cand
    return None


def text_files() -> List[str]:
    listed = []
    if os.path.exists(os.path.join(ROOT, ".git")):
        try:
            listed = subprocess.check_output(["git", "-C", ROOT, "ls-files", "-c", "-o", "--exclude-standard", "-z"],
                                             stderr=subprocess.DEVNULL, text=True).split("\0")
        except (OSError, subprocess.SubprocessError):
            pass
    if not listed:
        for base, dirs, names in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if base == ROOT and "models" in dirs:
                dirs.remove("models")
            listed.extend(os.path.relpath(os.path.join(base, n), ROOT).replace(os.sep, "/")
                          for n in names)
    return sorted(f for f in listed if f != SELF and f.endswith((".py", ".md"))
                  and not f.startswith("models/")
                  and not any(d in f.split("/") for d in SKIP_DIRS)
                  and os.path.isfile(os.path.join(ROOT, f)))


def bound_symbol(text: str, pos: int) -> Optional[str]:
    """The backticked identifier a citation is attached to, or None when it is attached to none."""
    matches = list(IDENT.finditer(text, 0, pos))
    if not matches:
        return None
    gap = text[matches[-1].end():pos]
    if len(gap) > GAP_LIMIT or gap.count("`") > 1:
        return None
    if any(word not in CONNECTORS for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', gap)):
        return None
    return matches[-1].group("name")


def cites_inside(lo: int, hi: int, named: Sequence["Symbol"]) -> bool:
    """Whether the cited range falls inside the symbol it names -- a fragment of a function is a
    legitimate citation, so the whole range has to sit between the definition's ends, not merely
    touch it: a range that starts above the ``def`` lands the reader on the function before it."""
    return any(s.start <= lo and hi <= s.end for s in named)


def positive_controls() -> None:
    """Prove the two detectors fire, against a symbol read from the tree rather than a literal.

    Without this, a checker that always returns "fine" passes exactly as loudly as one that works.
    The spans come from `laya_multimodal/models/base.py` as it is now, so the fixtures cannot themselves go stale.
    """
    fn = [s for s in symbols("laya_multimodal/models/base.py") if s.name == "_to_internal"]
    check_true("the control has a function to cite", bool(fn),
               "no `_to_internal` in laya_multimodal/models/base.py")
    if not fn:
        return
    first, last = fn[0].start, fn[0].end
    check_true("control: a range inside the named function is accepted",
               cites_inside(first, last, fn), "%d-%d rejected" % (first, last))
    check_true("control: a range 300 lines below it is rejected",
               not cites_inside(last + 1, last + 300, fn), "")
    check_true("control: line 1 of the cited file is rejected", not cites_inside(1, 1, fn), "")

    # The arms the overlap rule let through: a range that shares lines with the symbol but reaches
    # past either end lands the reader on whatever owns the line outside it.
    check_true("control: a range starting three lines above the def is rejected",
               not cites_inside(first - 3, last, fn), "%d-%d accepted" % (first - 3, last))
    check_true("control: a range ending three lines past the last line is rejected",
               not cites_inside(first, last + 3, fn), "%d-%d accepted" % (first, last + 3))
    check_true("control: a fragment of the function is still accepted",
               cites_inside(first + 1, first + 3, fn), "%d-%d rejected" % (first + 1, first + 3))

    stale = "`Agent._to_internal` (`laya_multimodal/models/base.py:%d`)" % (last + 300)
    got = bound_symbol(stale, stale.index("laya_multimodal/models/base.py"))
    check_true("control: the `symbol` (`file:NNN`) idiom binds",
               got == "Agent._to_internal", got)
    far = ("`Agent._to_internal` is the function that normalizes instructions before hashing, "
           "and it lives at (`laya_multimodal/models/base.py:%d`)" % last)
    got = bound_symbol(far, far.index("laya_multimodal/models/base.py"))
    check_true("control: a citation two clauses away does not bind", got is None, got)
    form = "`laya_multimodal/models/base.py::Agent._to_internal`"
    got = bound_symbol(form, form.index("Agent._to_internal"))
    check_true("control: a `::` citation is not read as a line citation", got is None, got)

    # The window itself, isolated: a gap of nothing but punctuation is still a gap of 31, and the
    # arm that only checks words-in-the-gap would bind here and call a distant symbol the subject.
    padded = "`Agent._to_internal`" + ("-" * 30) + "(`laya_multimodal/models/base.py:%d`)" % last
    got = bound_symbol(padded, padded.index("laya_multimodal/models/base.py"))
    check_true("control: a symbol 31 punctuation marks away is not the citation's subject",
               got is None, got)


def landing_page() -> None:
    """``docs/index.md`` is the site's front door, and it used to send three topics back to the
    README -- two of which have pages in this tree. A reader who trusts that sentence never finds
    them, so hold it to the tree: a topic the front door defers must have no page here, and every
    page it links must exist. The first half is the drift that actually happens (a page is added,
    the front door is not told); the second is the row this fix added."""
    path = os.path.join(ROOT, "docs", "index.md")
    text = open(path, encoding="utf-8").read()
    pages = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "docs")) if f.endswith(".md"))
    check_true("the landing-page check can see the guide pages", len(pages) >= 20,
               "only %d docs/*.md pages" % len(pages))

    for deferred in DEFERRAL.finditer(text):
        topics = [t.strip() for t in re.split(r",| and ", deferred.group("topics")) if t.strip()]
        check_true("the deferral lists the topics by name", len(topics) >= 1, topics)
        for topic in topics:
            slug = re.sub(r"\s+", "-", re.sub(r"^the\s+", "", topic.lower()).strip())
            check_true("the README-only claim about %r has no page in this tree" % topic,
                       slug not in pages,
                       "docs/%s.md exists, so the front door has to link it instead of deferring it"
                       % slug)

    links = [l for l in re.findall(r"\]\(([^)#\s]+\.md)\)", text) if not l.startswith("http")]
    broken = [l for l in links
              if not os.path.exists(os.path.join(ROOT, "docs", l.replace("/", os.sep)))]
    check_true("every guide the landing page links exists", not broken, broken)
    check_true("the landing-page link check has subjects", len(links) >= 10,
               "only %d relative links: %s" % (len(links), links))
    guides = {"multimodal.md", "multimodal-clients.md", "grounding.md", "multimodal-release.md"}
    check_true("the landing page links native guides", guides.issubset(links), sorted(guides.difference(links)))


def main() -> int:
    positive_controls()
    landing_page()
    files = text_files()
    expected = {"README.md", "docs/index.md", "laya_multimodal/models/multimodal.py", "scripts/multimodal/train.py",
                "examples/native_multimodal.py", "research/README.md", "benchmarks/README.md",
                "sdk/python/README.md", "sdk/typescript/README.md", "tests/test_hooks_api.py"}
    missing = sorted(expected.difference(files))
    check_true("the scan covers native code and guides", not missing, missing)

    dbl_total = 0
    dbl_owner_claims = 0
    unresolved: List[str] = []
    bound = 0
    drifted: List[str] = []
    unreadable: List[str] = []
    unbound = 0

    for rel in files:
        try:
            with open(os.path.join(ROOT, rel.replace("/", os.sep)), encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeError) as error:
            unreadable.append("%s: %s" % (rel, error))
            continue

        def where(pos: int, _rel: str = rel, _text: str = text) -> str:
            return "%s:%d" % (_rel, _text.count("\n", 0, pos) + 1)

        for m in DBLCOLON.finditer(text):
            tgt = target_of(m.group("file"))
            if tgt is None:
                continue
            dbl_total += 1
            owner, _, leaf = m.group("name").rpartition(".")
            named = [s for s in symbols(tgt) if s.name == leaf]
            if not named:
                unresolved.append("%s cites %s, which %s does not define"
                                  % (where(m.start()), m.group(0), tgt))
            elif owner:
                dbl_owner_claims += 1
                if not any(s.parent == owner for s in named):
                    unresolved.append("%s cites %s, but %s defines it as %s"
                                      % (where(m.start()), m.group(0), tgt, _shown(named)))

        for m in LINECITE.finditer(text):
            tgt = target_of(m.group("file"))
            if tgt is None:
                continue
            name = bound_symbol(text, m.start())
            if name is None:
                unbound += 1
                continue
            owner, _, leaf = name.rpartition(".")
            named = [s for s in symbols(tgt) if s.name == leaf]
            if not named:
                unbound += 1
                continue
            bound += 1
            if owner and not any(s.parent == owner for s in named):
                drifted.append("%s calls it %s, but %s defines it as %s"
                               % (where(m.start()), name, tgt, _shown(named)))
                continue
            lo, hi = int(m.group("a")), int(m.group("b") or m.group("a"))
            if not cites_inside(lo, hi, named):
                drifted.append("%s puts %s at %s:%d, but it runs %s"
                               % (where(m.start()), name, tgt, lo, _shown(named)))

    check_true("every discovered file can be read", not unreadable, unreadable)
    check_true("every `file::symbol` citation resolves to a definition",
               not unresolved, unresolved)
    check_true("the `file::symbol` check has subjects", dbl_total >= 12,
               "only %d citations of that form" % dbl_total)
    check_true("no `file:NNN` citation lands outside the symbol named beside it",
               not drifted, drifted)
    check_true("the line-citation check has subjects", bound >= 1,
               "only %d citations bind to a named symbol" % bound)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    print("%d text files scanned from %s" % (len(files), ROOT))
    print("%d `file::symbol` citation(s) resolve, %d of them claiming an owner"
          % (dbl_total, dbl_owner_claims))
    print("%d `file:NNN` citation(s) name a symbol beside them; %d of those land inside it"
          % (bound, bound - len(drifted)))
    print("not checkable from this repository (not asserted either way):")
    print("  - %d `file:NNN` citation(s) with no symbol named beside them -- a drift there is "
          "invisible, which is why the `::` form is the one to write" % unbound)
    print("  - citations to files outside the tree, and to non-Python targets")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
