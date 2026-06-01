#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
|                     DEAN PROJECT AUDITOR  v1.0                             |
|     Python-      |
--------------------------------------------------------------------------------

:
    python audit.py                     #   
    python audit.py --root src          #  
    python audit.py --root src --json   #   JSON
    python audit.py --root src --fix-hints  #   

 :
  [ARC]     , tight coupling, God Objects
  [DUP]     , copy-paste,  
  [BUG]     async, exception handling, None-checks
  [CFG]    hardcoded ,  
  [SEC]    SQL injection, credentials,  
  [CMX]    cyclomatic complexity,  /
  [TYP]     ,   
  [LOG]    , ,  
  [IMP]    , ,  
  [ML]   ML-   , noise filter, feature consistency
"""

import ast
import collections
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --   -----------------------------------------------------

SEVERITY = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
SEVERITY_EMOJI = {"CRITICAL": "[CRITICAL]", "HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]", "INFO": "[INFO]"}

#    DEAN
DEAN_RULES = {
    # Hardcoded       ModelFactory
    "hardcoded_model_list": re.compile(
        r"[\[\(]\s*['\"](:lightgbm|xgboost|catboost|lstm|gru|transformer|random_forest|svm|knn)['\"]"
        r"(:\s*,\s*['\"](:lightgbm|xgboost|catboost|lstm|gru|transformer|random_forest|svm|knn)['\"])+\s*[\]\)]"
    ),
    #  noise filter    0.5 * rolling_std
    "fixed_noise_filter": re.compile(
        r"noise\s*[+\-]=\s*[\d.]+(!\s*\*\s*rolling)"
    ),
    # ExperienceDiaryEngine   
    "deprecated_diary": re.compile(r"ExperienceDiaryEngine"),
    # DiaryEngine.record   
    "diary_correct": re.compile(r"DiaryEngine\.record_decision\(DecisionRecord"),
    # 60/40 split   
    "wrong_knn_split": re.compile(r"(:0\.[1-9][0-9]*\s*\*\s*knn|knn\s*\*\s*0\.[1-9][0-9]*)"),
    # asyncio.run()  async context
    "asyncio_run_in_async": re.compile(r"asyncio\.run\("),
}

SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".pytest_cache",
    "dist", "build", "*.egg-info",
}


# --   -----------------------------------------------------------

@dataclass
class Issue:
    category:  str
    severity:  str
    file:      str
    line:      int
    message:   str
    code:      str = ""
    fix_hint:  str = ""

    def __str__(self) -> str:
        emoji = SEVERITY_EMOJI.get(self.severity, "[INFO]")
        loc   = f"{self.file}:{self.line}"
        parts = [f"{emoji} [{self.severity}] [{self.category}] {loc}"]
        parts.append(f"   {self.message}")
        if self.code:
            parts.append(f"   code: {self.code.strip()[:120]}")
        if self.fix_hint:
            parts.append(f"   fix:  {self.fix_hint}")
        return "\n".join(parts)


@dataclass
class AuditResult:
    issues:        list[Issue] = field(default_factory=list)
    stats:         dict[str, Any] = field(default_factory=dict)
    file_count:    int = 0
    line_count:    int = 0
    import_graph:  dict[str, set[str]] = field(default_factory=lambda: collections.defaultdict(set))

    def add(self, *args, **kwargs) -> None:
        self.issues.append(Issue(*args, **kwargs))

    def summary(self) -> dict[str, int]:
        cnt: dict[str, int] = collections.Counter()
        for iss in self.issues:
            cnt[iss.severity] += 1
        return dict(cnt)


# --   ---------------------------------------------------------

def iter_py_files(root: Path):
    # Optimized: only traverse relevant source directories
    target_dirs = ['src', 'scripts']
    for t_dir in target_dirs:
        d = root / t_dir
        if not d.exists(): continue
        for path in d.rglob("*.py"):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            yield path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_file(path: Path) -> tuple[str, list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, text.splitlines()
    except Exception:
        return "", []


def parse_ast(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def module_name(path: Path, root: Path) -> str:
    """Convert file path to dotted module name."""
    try:
        rel_path = path.relative_to(root)
        parts = list(rel_path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    except ValueError:
        return path.stem


def cyclomatic_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Simplified McCabe complexity."""
    cc = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                               ast.With, ast.Assert, ast.comprehension)):
            cc += 1
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
    return cc


# --  --------------------------------------------------------------------

class ImportChecker:
    """[IMP]  [ARC]  ,  ."""

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)
        mod   = module_name(path, root)

        imports_used: set[str] = set()
        imports_defined: list[tuple[int, str]] = []

        #   
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports_defined.append((node.lineno, name))
                    result.import_graph[mod].add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                mod_from = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        result.add("IMP", "MEDIUM", fname, node.lineno,
                                   f"Wildcard import: from {mod_from} import *",
                                   fix_hint=f"    {mod_from}")
                    name = alias.asname or alias.name
                    imports_defined.append((node.lineno, name))
                    result.import_graph[mod].add(mod_from.split(".")[0] if mod_from else "")

        #    
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                imports_used.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    imports_used.add(node.value.id)

        #  
        for lineno, name in imports_defined:
            if name not in imports_used and name != "_":
                result.add("IMP", "LOW", fname, lineno,
                           f"  : {name}",
                           lines[lineno - 1] if lineno <= len(lines) else "",
                           fix_hint="   # noqa: F401")


class ComplexityChecker:
    """[CMX]  ,  /."""

    CC_THRESHOLD   = 12   # cyclomatic complexity
    FUNC_LINES     = 60   #   
    CLASS_LINES    = 400  #   
    CLASS_METHODS  = 25   #   
    CLASS_ATTRS    = 20   #   __init__

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):
            # 
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = cyclomatic_complexity(node)
                if cc > self.CC_THRESHOLD:
                    result.add("CMX", "HIGH", fname, node.lineno,
                               f" cyclomatic complexity: {node.name}() = {cc} "
                               f"( {self.CC_THRESHOLD})",
                               fix_hint="    / use early returns")

                func_lines = (node.end_lineno or node.lineno) - node.lineno
                if func_lines > self.FUNC_LINES:
                    result.add("CMX", "MEDIUM", fname, node.lineno,
                               f" : {node.name}() = {func_lines}  "
                               f"( {self.FUNC_LINES})",
                               fix_hint="   / dataclass")

            # 
            elif isinstance(node, ast.ClassDef):
                class_lines = (node.end_lineno or node.lineno) - node.lineno
                if class_lines > self.CLASS_LINES:
                    result.add("CMX", "HIGH", fname, node.lineno,
                               f"God Class: {node.name} = {class_lines}  "
                               f"( {self.CLASS_LINES})",
                               fix_hint="    /  ")

                methods = [n for n in ast.walk(node)
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and n.col_offset > node.col_offset]
                if len(methods) > self.CLASS_METHODS:
                    result.add("CMX", "MEDIUM", fname, node.lineno,
                               f"  : {node.name} = {len(methods)} "
                               f"( {self.CLASS_METHODS})",
                               fix_hint="      / mixins")

                #    __init__
                for method in methods:
                    if method.name == "__init__":
                        assigns = [n for n in ast.walk(method)
                                   if isinstance(n, ast.Assign)
                                   and any(
                                       isinstance(t, ast.Attribute)
                                       and isinstance(t.value, ast.Name)
                                       and t.value.id == "self"
                                       for t in n.targets
                                   )]
                        if len(assigns) > self.CLASS_ATTRS:
                            result.add("CMX", "MEDIUM", fname, method.lineno,
                                       f"{node.name}.__init__  {len(assigns)}  "
                                       f"( {self.CLASS_ATTRS})   God Object",
                                       fix_hint="   dataclass   ")


class BugChecker:
    """[BUG]   ."""

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname  = rel(path, root)
        is_async_file = any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(tree))

        for node in ast.walk(tree):

            # asyncio.run()  async-
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call) and
                            isinstance(child.func, ast.Attribute) and
                            isinstance(child.func.value, ast.Name) and
                            child.func.value.id == "asyncio" and
                            child.func.attr == "run"):
                        line = lines[child.lineno - 1] if child.lineno <= len(lines) else ""
                        result.add("BUG", "CRITICAL", fname, child.lineno,
                                   "asyncio.run()  async-   event loop",
                                   line,
                                   fix_hint="  `await coroutine`  `asyncio.ensure_future()`")

            # Broad except  re-raise
            if isinstance(node, ast.ExceptHandler):
                if node.type is None or (
                    isinstance(node.type, ast.Name) and node.type.id == "Exception"
                ):
                    body_raises = any(isinstance(n, ast.Raise) for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
                    body_logs   = any(
                        isinstance(n, ast.Call) and
                        isinstance(getattr(n.func, 'attr', None), str) and
                        n.func.attr in ("error", "critical", "exception")
                        for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
                    )
                    if not body_raises and not body_logs:
                        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        result.add("BUG", "HIGH", fname, node.lineno,
                                   "Broad except     re-raise    ",
                                   line,
                                   fix_hint=" logger.error(exc)  re-raise")

            #   None  ==  is
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Eq, ast.NotEq)) and (
                        isinstance(comp, ast.Constant) and comp.value is None
                    ):
                        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        result.add("BUG", "LOW", fname, node.lineno,
                                   " `== None`  `is None`",
                                   line,
                                   fix_hint=" `== None`  `is None`")

            # Mutable default arguments
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        result.add("BUG", "HIGH", fname, node.lineno,
                                   f"{node.name}()  mutable default argument (list/dict/set)",
                                   fix_hint=" `def f(x=[]):`  `def f(x=None): x = x or []`")

            # Return  async-  await
            if isinstance(node, ast.AsyncFunctionDef):
                has_await = any(isinstance(n, ast.Await) for n in ast.walk(node))
                if not has_await and node.name not in ("__aenter__", "__aexit__"):
                    result.add("BUG", "LOW", fname, node.lineno,
                               f"async def {node.name}()    `await`    `async`",
                               fix_hint=f" `async`     coroutine")


class DuplicateChecker:
    """[DUP]     ."""

    MIN_BLOCK_LINES = 8  #     

    def __init__(self):
        self._blocks: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)

    def collect(self, path: Path, root: Path, lines: list[str]) -> None:
        fname = rel(path, root)
        #   (   )
        for start in range(len(lines) - self.MIN_BLOCK_LINES):
            block = lines[start:start + self.MIN_BLOCK_LINES]
            #        /  
            code_lines = [l.strip() for l in block if l.strip() and not l.strip().startswith("#")]
            if len(code_lines) < self.MIN_BLOCK_LINES // 2:
                continue
            normalized = "\n".join(re.sub(r'\s+', ' ', l) for l in code_lines)
            key = hashlib.md5(normalized.encode()).hexdigest()
            self._blocks[key].append((fname, start + 1))

    def report(self, result: AuditResult) -> None:
        for key, locations in self._blocks.items():
            if len(locations) >= 2:
                #     
                unique_files = list({loc[0] for loc in locations})
                if len(unique_files) >= 2:
                    files_str = ", ".join(f"{f}:{l}" for f, l in locations[:4])
                    result.add("DUP", "MEDIUM", locations[0][0], locations[0][1],
                               f"   {len(unique_files)} : {files_str}",
                               fix_hint="   /")


class SecurityChecker:
    """[SEC]  ."""

    #   f-string,   SQL-,    logger/Path
    SQL_PATTERN = re.compile(
        r'\.execute\s*\(\s*f["\']', 
        re.IGNORECASE
    )
    SECRET_NAMES = re.compile(r'(:password|secret|api_key|token|credential|private_key)\s*=\s*["\'][^"\']{4,}["\']', re.IGNORECASE)
    PATH_INJECT  = re.compile(r'open\s*\(\s*f["\']')

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)
        for i, line in enumerate(lines, 1):
            if self.SQL_PATTERN.search(line):
                result.add("SEC", "HIGH", fname, i,
                           " SQL injection  f-string",
                           line.strip(),
                           fix_hint="  : con.execute(sql, params)")

            if self.SECRET_NAMES.search(line):
                result.add("SEC", "CRITICAL", fname, i,
                           " hardcoded secret/password  ",
                           re.sub(r'=\s*["\'][^"\']+["\']', '= "***"', line.strip()),
                           fix_hint="  .env / SecretsManager")

            if self.PATH_INJECT.search(line):
                result.add("SEC", "MEDIUM", fname, i,
                           "open()  f-string path   path traversal",
                           line.strip(),
                           fix_hint=" Path().resolve()       ")


class MLChecker:
    """[ML]  ML-   DEAN."""

    #   
    LEAKAGE_PATTERNS = [
        (re.compile(r'fit\(.*test', re.IGNORECASE),     "fit()  test   data leakage"),
        (re.compile(r'fit_transform\(.*test', re.IGNORECASE), "fit_transform()  test  data leakage"),
        (re.compile(r'StandardScaler\(\)\.fit\('), "Scaler.fit()  .transform()   "),
    ]

    #  noise  
    FIXED_NOISE = re.compile(r'noise\s*=\s*(!0\.5\s*\*)[0-9.]+')

    # Hardcoded feature counts
    HARDCODED_FEATURES = re.compile(r'(:n_features|num_features|feature_count)\s*=\s*(!20|15|50|30)[0-9]+')

    # Hardcoded 60/40 split violation
    WRONG_SPLIT = re.compile(r'(:model_weight|knn_weight)\s*=\s*(!0\.6|0\.4)[0-9.]+')

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Data leakage
            for pattern, msg in self.LEAKAGE_PATTERNS:
                if pattern.search(line):
                    result.add("ML", "CRITICAL", fname, i, msg, stripped,
                               fix_hint="fit()   train split, transform()  test")

            #  noise filter
            if self.FIXED_NOISE.search(line) and "rolling_std" not in line:
                result.add("ML", "HIGH", fname, i,
                           " noise filter     (0.5 * rolling_std)",
                           stripped,
                           fix_hint="noise = 0.5 * df['close'].rolling(window).std()")

            # Hardcoded model list (  DEAN)
            if DEAN_RULES["hardcoded_model_list"].search(line):
                result.add("ML", "HIGH", fname, i,
                           "Hardcoded     canonical source (ModelFactory)",
                           stripped,
                           fix_hint=" ModelFactory.get_available_models()  get_light_models()")

            #  ExperienceDiaryEngine
            if DEAN_RULES["deprecated_diary"].search(line):
                result.add("ML", "CRITICAL", fname, i,
                           "ExperienceDiaryEngine    DiaryEngine.record_decision(DecisionRecord)",
                           stripped,
                           fix_hint="from src.core.diary import DiaryEngine, DecisionRecord")

            # Wrong split
            if self.WRONG_SPLIT.search(line):
                result.add("ML", "MEDIUM", fname, i,
                           "   60/40 split (model/KNN)",
                           stripped,
                           fix_hint="model_weight=0.6, knn_weight=0.4   ")


class HardcodeChecker:
    """[CFG]  hardcoded , magic numbers."""

    MAGIC_NUMBER = re.compile(r'(<![0-9.])(:0\.[1-9][0-9]*|[1-9][0-9]{2,})(![0-9.])')
    SKIP_CONTEXTS = {"__version__", "lineno", "maxsize", "timeout", "port", "status_code"}

    #  
    HARDCODED_PATH = re.compile(r'["\'](:/home/|/Users/|C:\\\\|D:\\\\|/tmp/)[^"\']+["\']')

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            # Hardcoded paths
            if self.HARDCODED_PATH.search(line):
                result.add("CFG", "MEDIUM", fname, i,
                           "Hardcoded    ",
                           stripped,
                           fix_hint=" config_manager.get_config('paths.xxx')  Path(__file__).parent")

            #  URL/IP
            if re.search(r'["\'](:http://|https://)[^"\']+["\']', line) and "test" not in fname.lower():
                result.add("CFG", "LOW", fname, i,
                           "Hardcoded URL    ",
                           stripped)


class TypeChecker:
    """[TYP]     ."""

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):
            #   return type annotation
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") or node.name == "__init__":
                    continue
                if node.returns is None and len(node.body) > 3:
                    result.add("TYP", "LOW", fname, node.lineno,
                               f" return type annotation: {node.name}()",
                               fix_hint=f"def {node.name}(...) -> ReturnType:")

            # isinstance  tuple  Union
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and node.func.id == "isinstance"
                        and len(node.args) == 2):
                    if isinstance(node.args[1], ast.Tuple) and len(node.args[1].elts) > 4:
                        result.add("TYP", "LOW", fname, node.lineno,
                                   f"isinstance  {len(node.args[1].elts)}    Union[...]",
                                   fix_hint="from typing import Union; Union[A, B, C]")


class CouplingChecker:
    """[ARC]  tight coupling, circular deps."""

    def find_cycles(self, graph: dict[str, set[str]], result: AuditResult) -> None:
        """      DFS."""
        visited:    set[str] = set()
        rec_stack: set[str] = set()
        cycles:    list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, set()):
                #   
                if not any(neighbor.startswith(p) for p in ("src.", "pipeline.", "core.", "models.", "data.", "training.")):
                    continue
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor])
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor) if neighbor in path else 0
                    cycle = path[cycle_start:] + [neighbor]
                    if len(cycle) >= 2:
                        cycles.append(cycle)

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node, [node])

        seen_cycles: set[str] = set()
        for cycle in cycles:
            key = "".join(sorted(cycle))
            if key in seen_cycles:
                continue
            seen_cycles.add(key)
            cycle_str = "  ".join(cycle)
            result.add("ARC", "CRITICAL", cycle[0].replace(".", "/") + ".py", 1,
                       f" : {cycle_str}",
                       fix_hint=" lazy import (TYPE_CHECKING)   ")

    def check_fan_out(self, path: Path, root: Path, tree: ast.Module, result: AuditResult) -> None:
        """      (fan-out)."""
        fname = rel(path, root)
        internal_imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("src."):
                    internal_imports.add(node.module)

        if len(internal_imports) > 15:
            result.add("ARC", "HIGH", fname, 1,
                       f"Tight coupling: {len(internal_imports)}   "
                       f"( 15)   God Module",
                       fix_hint="   facade  dependency injection")


class LoggingChecker:
    """[LOG]    ."""

    def run(self, path: Path, root: Path, text: str, lines: list[str],
            tree: ast.Module, result: AuditResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                #   100    logger.*
                class_text = "\n".join(
                    lines[node.lineno - 1:(node.end_lineno or node.lineno + 100)]
                )
                class_lines = (node.end_lineno or node.lineno) - node.lineno
                if class_lines > 100 and "logger" not in class_text and "logging" not in class_text:
                    result.add("LOG", "LOW", fname, node.lineno,
                               f" {node.name} ({class_lines} )  ",
                               fix_hint=" self.logger = ProjectLogger.get_logger(__name__)")

            # print()  production 
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if "debug" not in fname.lower() and "test" not in fname.lower():
                        result.add("LOG", "LOW", fname, node.lineno,
                                   "print()  production    logger",
                                   line.strip(),
                                   fix_hint="logger.debug(msg)  logger.info(msg)")


# --   ----------------------------------------------------------

class ProjectAuditor:

    def __init__(self, root: Path) -> None:
        self.root        = root
        self.result      = AuditResult()
        self.dup_checker = DuplicateChecker()
        self.checkers    = [
            ImportChecker(),
            ComplexityChecker(),
            BugChecker(),
            SecurityChecker(),
            MLChecker(),
            HardcodeChecker(),
            TypeChecker(),
            LoggingChecker(),
        ]
        self.coupling_checker = CouplingChecker()

    def audit(self) -> AuditResult:
        print(f"SEARCH: : {self.root}")
        files = list(iter_py_files(self.root))
        self.result.file_count = len(files)

        for path in files:
            text, lines = read_file(path)
            self.result.line_count += len(lines)
            if not text:
                continue

            tree = parse_ast(text)
            if tree is None:
                self.result.add("BUG", "HIGH", rel(path, self.root), 1,
                                "SyntaxError    ",
                                fix_hint="  ")
                continue

            #   
            for checker in self.checkers:
                checker.run(path, self.root, text, lines, tree, self.result)

            #   
            self.dup_checker.collect(path, self.root, lines)

            # Fan-out 
            self.coupling_checker.check_fan_out(path, self.root, tree, self.result)

        #  
        self.dup_checker.report(self.result)

        #    
        self.coupling_checker.find_cycles(
            dict(self.result.import_graph), self.result
        )

        # 
        self.result.stats = {
            "files":  self.result.file_count,
            "lines":  self.result.line_count,
            "issues": len(self.result.issues),
            "by_severity": self.result.summary(),
            "by_category": dict(
                collections.Counter(i.category for i in self.result.issues)
            ),
        }

        return self.result


# --  ---------------------------------------------------------------------

def print_report(result: AuditResult, show_fix_hints: bool = True, max_issues: int = 500) -> None:
    sev_order = list(SEVERITY.keys())

    issues_sorted = sorted(
        result.issues,
        key=lambda i: (SEVERITY.get(i.severity, 99), i.category, i.file, i.line),
    )

    current_sev = None
    shown = 0
    for issue in issues_sorted:
        if shown >= max_issues:
            print(f"\n...   {len(issues_sorted) - shown} issues ( --json  )")
            break
        if issue.severity != current_sev:
            current_sev = issue.severity
            print(f"\n{'-' * 70}")
            print(f"  {SEVERITY_EMOJI[issue.severity]} {issue.severity}")
            print(f"{'-' * 70}")

        emoji = SEVERITY_EMOJI.get(issue.severity, "[INFO]")
        print(f"\n{emoji} [{issue.category}] {issue.file}:{issue.line}")
        print(f"   {issue.message}")
        if issue.code:
            print(f"   > {issue.code.strip()[:110]}")
        if show_fix_hints and issue.fix_hint:
            print(f"   - {issue.fix_hint}")
        shown += 1

    # 
    s = result.stats
    print(f"\n{'-' * 70}")
    print(f"  STATS:  ")
    print(f"{'-' * 70}")
    print(f"    : {s['files']}")
    print(f"              : {s['lines']:,}")
    print(f"   issues         : {s['issues']}")
    print()
    for sev in sev_order:
        cnt = s['by_severity'].get(sev, 0)
        if cnt:
            print(f"  {SEVERITY_EMOJI[sev]} {sev:<12}: {cnt}")
    print()
    print("   :")
    for cat, cnt in sorted(s['by_category'].items(), key=lambda x: -x[1]):
        print(f"    [{cat}] {cnt}")
    print(f"{'-' * 70}")


# -- entry point ---------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DEAN Project Auditor     Python-"
    )
    parser.add_argument("--root",       default=".",     help="  (default: .)")
    parser.add_argument("--json",       action="store_true", help="  JSON")
    parser.add_argument("--fix-hints",  action="store_true", default=True, help="   ")
    parser.add_argument("--severity",   default="LOW",   help=" : CRITICAL/HIGH/MEDIUM/LOW/INFO")
    parser.add_argument("--category",   default="",      help="  : ARC,DUP,BUG,CFG,SEC,CMX,TYP,LOG,IMP,ML")
    parser.add_argument("--output",     default="",      help="   ")
    parser.add_argument("--max-issues", default=500, type=int, help=" issues  ")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR:   : {root}")
        sys.exit(1)

    auditor = ProjectAuditor(root)
    result  = auditor.audit()

    # 
    min_sev = SEVERITY.get(args.severity.upper(), 3)
    result.issues = [
        i for i in result.issues
        if SEVERITY.get(i.severity, 99) <= min_sev
        and (not args.category or i.category in args.category.upper().split(","))
    ]

    if args.json:
        output = json.dumps(
            {
                "stats":  result.stats,
                "issues": [
                    {
                        "category": i.category,
                        "severity": i.severity,
                        "file":     i.file,
                        "line":     i.line,
                        "message":  i.message,
                        "fix_hint": i.fix_hint,
                    }
                    for i in result.issues
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"DONE: : {args.output}")
        else:
            print(output)
    else:
        #     
        if args.output:
            import contextlib
            with open(args.output, "w", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f):
                    print_report(result, args.fix_hints, args.max_issues)
            print(f"DONE: : {args.output}")
        else:
            print_report(result, args.fix_hints, args.max_issues)

    # Exit code   severity
    critical = result.stats.get("by_severity", {}).get("CRITICAL", 0)
    high     = result.stats.get("by_severity", {}).get("HIGH", 0)
    sys.exit(2 if critical > 0 else (1 if high > 0 else 0))


if __name__ == "__main__":
    main()
