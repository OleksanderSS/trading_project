"""Find config keys that are read, sometimes logged, and never used.

A dead config key is worse than a missing one. A missing key fails loudly the
first time something looks for it. A dead key sits in the YAML looking like a
control, gets read into an attribute, often gets printed in a startup line --
and decides nothing. Editing it changes the log and nothing else.

Three of these have cost real time here:

  `daily_max_years: 2`  sat in the yahoo_finance block with zero readers and
                        read like the live limit. Cost an afternoon, and the
                        note left behind says "removed rather than wired,
                        because two places declaring one number is how every
                        fix in this project has half-landed".
  `vix.params.period`   was read into `self.period`, printed in
                        "VIXCollector initialized. Period: 30d", and then the
                        fetch said `history(period="60d")` outright. The
                        config declared 30, sixty were collected, and neither
                        number had been chosen by anyone.
  `attention_window`    a 20 in `TickerExternalEnricher.__init__` that no
                        config file set and no line read.

The scan runs in two passes because one pass over-reports badly.

  Pass 1 finds `self.x = <something>.get(...)` where the source mentions
  config/param/setting, then checks every load of `self.x` INSIDE the class.
  This alone flags `table_name` and `timeout` on half the collectors, because
  those are read by the BASE class, not the subclass that assigns them.

  Pass 2 therefore looks for `.x` across the whole of `src/`, skipping the
  assignment itself and any line that logs. Only attributes with no reader
  anywhere survive. On 2026-08-22 that took 39 candidates down to 22.

Loads inside logging calls do not count as use. That is the whole point: the
VIX period was "used" in exactly that sense.

Runnable standalone:  python tests/contracts/_dead_config_scan.py
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"

EXCLUDED_PARTS = ("archive", "draft", "__pycache__", ".venv", "venv")

#: Names that mark a call as logging rather than use.
_LOGGING_HINTS = ("logger", "log", "print", "debug", "info", "warning",
                  "error", "exception", "critical")

#: What the object being `.get()` from has to look like for the read to count
#: as a configuration read.
_CONFIG_HINTS = ("config", "param", "setting", "opts", "options")


@dataclass(frozen=True)
class Finding:
    path: str
    cls: str
    attribute: str
    line: int
    logged: int

    def __str__(self) -> str:
        where = f"logged {self.logged}x" if self.logged else "never read at all"
        return f"{self.path}:{self.line} {self.cls}.{self.attribute} [{where}]"


def _config_read(node: ast.AST) -> bool:
    """True when the expression pulls a value out of a config-like mapping."""
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"):
            source = ast.unparse(sub.func.value).lower()
            if any(hint in source for hint in _CONFIG_HINTS):
                return True
    return False


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    mapping: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[child] = parent
    return mapping


def _inside_logging(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Call) and hasattr(current, "func"):
            text = ast.unparse(current.func).lower()
            if any(hint in text for hint in _LOGGING_HINTS):
                return True
        current = parents.get(current)
    return False


@lru_cache(maxsize=1)
def _sources() -> tuple[tuple[str, str], ...]:
    collected = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        collected.append((str(path), path.read_text(encoding="utf-8", errors="replace")))
    return tuple(collected)


def _read_anywhere(attribute: str) -> bool:
    """True when `.attribute` is read somewhere that is not an assignment or a log."""
    pattern = re.compile(rf"\.{re.escape(attribute)}\b")
    assignment = re.compile(rf"self\.{re.escape(attribute)}\s*=")
    for _, text in _sources():
        for match in pattern.finditer(text):
            start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            line = text[start: end if end > 0 else len(text)]
            if assignment.search(line):
                continue
            if any(hint in line.lower() for hint in ("logger", "log.", "print(")):
                continue
            return True
    return False


def scan() -> list[Finding]:
    """Config-derived attributes with no reader anywhere in src/."""
    findings: list[Finding] = []
    for path, text in _sources():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        parents = _parents(tree)
        relative = Path(path).relative_to(PROJECT_ROOT).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue

            assigned: dict[str, int] = {}
            for node in ast.walk(cls):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if not (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    continue
                if _config_read(node.value):
                    assigned.setdefault(target.attr, node.lineno)

            if not assigned:
                continue

            used = dict.fromkeys(assigned, 0)
            logged = dict.fromkeys(assigned, 0)
            for node in ast.walk(cls):
                if not (isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                        and node.attr in assigned):
                    continue
                if _inside_logging(node, parents):
                    logged[node.attr] += 1
                else:
                    used[node.attr] += 1

            for attribute, line in assigned.items():
                if used[attribute] == 0 and not _read_anywhere(attribute):
                    findings.append(
                        Finding(relative, cls.name, attribute, line, logged[attribute])
                    )
    return sorted(findings, key=lambda finding: (finding.path, finding.line))


# ---------------------------------------------------------------------------
# The third pass, added 2026-09-07.
#
# Everything above finds a key that IS read into an attribute and then goes
# unused. It is blind to the other half of the family: a key no Python line
# ever mentions. `huggingface.max_rows: 10000` is declared beside a collector
# that loads the entire split, and the name `max_rows` does not occur anywhere
# in src/ -- so nothing above can see it, and the table holds 999,396 rows.
#
# Worse cases found by the same query on the day it was written:
#
#   `cache_duration_minutes`  declared in SEVENTEEN collector blocks. Its twin
#                             `cache_ttl` sits beside it in seconds and IS read
#                             by BaseCollector.get_cache_ttl. Editing the
#                             readable one in minutes changes nothing.
#   `intraday_max_days: 60`   in the yahoo_finance block. That single 60 is the
#                             exact defect the code removed when
#                             _INTRADAY_HISTORY_LIMIT_DAYS was introduced,
#                             which recovered 92% of the hourly history. The
#                             config still declares the broken number.
#
# A leaf is only reported when NEITHER its own name NOR any ancestor key below
# the top level occurs as a quoted string in src/. The ancestor rule is what
# keeps `headers.Content-Type`, `column_mapping.trade_date` and
# `indicators.IMF` out: those dicts are passed and iterated whole, so the sub
# keys are reached without ever being named.
# ---------------------------------------------------------------------------

#: Config files this pass reads. Adding one raises the count, so add it in the
#: same commit that lowers the ceiling to whatever it then measures.
CONFIGS_SCANNED = ("src/config/collectors.yaml",)


@dataclass(frozen=True)
class UnreadKey:
    config: str
    path: str

    def __str__(self) -> str:
        return f"{self.config}: {self.path}"


def _quoted_anywhere(name: str) -> bool:
    """Does `name` appear as a quoted string in any non-archive source file?"""
    pattern = re.compile(r"['\"]" + re.escape(name) + r"['\"]")
    return any(pattern.search(text) for _, text in _sources())


def _mentioned_anywhere(name: str) -> bool:
    """Does `name` appear as a bare identifier anywhere in src/?

    Deliberately looser than `_quoted_anywhere`, and used only for ancestors.
    """
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    return any(pattern.search(text) for _, text in _sources())


def _leaves(node, trail=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, trail + (str(key),))
    else:
        yield trail


def scan_never_read() -> list[UnreadKey]:
    """YAML keys whose name no source file mentions, directly or by ancestor."""
    import yaml

    findings: list[UnreadKey] = []
    for relative in CONFIGS_SCANNED:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for trail in _leaves(document):
            # trail[0] is the file's own top section, trail[1] the collector
            # name; neither is a setting. A one-key trail is not a leaf worth
            # judging.
            if len(trail) < 3:
                continue
            # The leaf must be named in quotes to count as read. An ANCESTOR
            # need only appear as a bare word: `headers=self.configs[...]`
            # passes the whole dict, and its sub keys are then reached without
            # any of them ever being written down.
            if _quoted_anywhere(trail[-1]):
                continue
            if any(_mentioned_anywhere(name) for name in trail[2:-1]):
                continue
            findings.append(UnreadKey(relative, ".".join(trail)))
    return sorted(findings, key=str)


# ---------------------------------------------------------------------------
# The MIRROR, added 2026-09-08.
#
# The two passes above find a config key with no reader. This one finds the
# opposite: a reader with no config key. `configs.get('market_impact_
# coefficient', 0.1)` where no yaml declares that name means the 0.1 wins every
# time, in every block -- and that particular 0.1 drove 74% of the cost of a
# $25,000 order while being written down nowhere an operator would look
# (#290, R66).
#
# #288 left this unautomated with a stated reason: "there is no reliable static
# way to determine which config.get(key) reads THIS block", and two attempts
# missed and over-reached in opposite directions. That reason is sound and this
# does not argue with it -- it takes the SUBSET where the question does not
# arise. If the key appears in no yaml at all, it does not matter which block
# the call reads: none of them declares it, so the default always governs.
#
# Numeric defaults only. A string or dict default is usually a name or a shape;
# a number is a policy somebody would want to set.
# ---------------------------------------------------------------------------

#: What the object being `.get()` from must look like. Wider than _CONFIG_HINTS
#: above because this pass reads the call site, not an assignment.
_GETTER_HINTS = ("config", "configs", "params", "settings", "opts", "cfg")


@dataclass(frozen=True)
class UndeclaredDefault:
    path: str
    line: int
    key: str
    default: float

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.key} = {self.default}"


@lru_cache(maxsize=1)
def _declared_yaml_keys() -> frozenset[str]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (PROJECT_ROOT / "src" / "config").glob("*.yaml"))
    return frozenset(
        re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", text, re.M))


def _numeric_constant(node: ast.AST) -> float | None:
    negative = False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node, negative = node.operand, True
    if not (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return None
    return -node.value if negative else node.value


def scan_undeclared_defaults() -> list[UndeclaredDefault]:
    """Numeric `.get(key, number)` defaults whose key no yaml declares.

    Two shapes are excluded, and both were found by reading the first run's
    output rather than trusting its count:

    A NESTED FALLBACK. `get('max_position_size_pct', get('max_position_size',
    0.1))` reads a DECLARED key and falls back to an old name. The inner call
    is not what governs, and flagging it would report the very pattern #288
    introduced deliberately so an existing deployment does not lose its
    setting. A `.get` sitting in the default slot of another `.get` is skipped.

    A DOTTED PATH. `config.get('strategy.risk_management.anomaly_threshold',
    0.8)` asks a config manager to walk a path; no yaml line is ever literally
    that string. The last segment is what a yaml declares, so that is what is
    checked.
    """
    declared = _declared_yaml_keys()
    findings: list[UndeclaredDefault] = []
    for path, text in _sources():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        relative = Path(path).relative_to(PROJECT_ROOT).as_posix()

        # Every `.get` that is the default of another `.get`: a fallback.
        fallbacks = {
            id(inner)
            for outer in ast.walk(tree)
            if isinstance(outer, ast.Call) and len(outer.args) == 2
            for inner in [outer.args[1]]
            if isinstance(inner, ast.Call)
        }
        # And every `.get` in the else-branch of `if '<declared key>' in
        # config:`. Same fallback in statement form -- the engine's spread is
        # written that way, reading a declared `spread_pct` and dropping to
        # `spread_bps` only when it is absent.
        for branch in ast.walk(tree):
            if not isinstance(branch, ast.If):
                continue
            test = branch.test
            if not (isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.In)
                    and isinstance(test.left, ast.Constant)
                    and isinstance(test.left.value, str)
                    and test.left.value.rsplit(".", 1)[-1] in declared):
                continue
            for statement in branch.orelse:
                for inner in ast.walk(statement):
                    if isinstance(inner, ast.Call):
                        fallbacks.add(id(inner))

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2):
                continue
            if id(node) in fallbacks:
                continue
            if not any(hint in ast.unparse(node.func.value).lower()
                       for hint in _GETTER_HINTS):
                continue
            key = node.args[0]
            if not (isinstance(key, ast.Constant)
                    and isinstance(key.value, str)):
                continue
            # A dotted path is walked by the config manager; the last segment
            # is the name a yaml line actually carries.
            if key.value.rsplit(".", 1)[-1] in declared:
                continue
            number = _numeric_constant(node.args[1])
            if number is None:
                continue
            findings.append(
                UndeclaredDefault(relative, node.lineno, key.value, number))
    return sorted(findings, key=lambda f: (f.path, f.line))


if __name__ == "__main__":
    results = scan()
    for finding in results:
        print(finding)
    print(f"\n{len(results)} config keys that decide nothing")
    unread = scan_never_read()
    for key in unread:
        print(key)
    print(f"\n{len(unread)} config keys no source file mentions at all")
    undeclared = scan_undeclared_defaults()
    for finding in undeclared:
        print(finding)
    print(f"\n{len(undeclared)} numbers that govern from code, declared in no "
          f"yaml")
