#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
|              DEAN DEEP LOGIC AUDITOR  v1.0                                 |
|   ,  ,  , ML-       |
--------------------------------------------------------------------------------

:
    python audit_logic.py --root src
    python audit_logic.py --root src --json --output logic_report.json
    python audit_logic.py --root src --category ML,LEAK,PANDAS

  (   audit.py):
  [LEAK]     traintest, temporal leakage, target leakage
  [PANDAS] - pandas: chained indexing, view vs copy, NaN
  [LOGIC]   : dead code,  , off-by-one
  [ASYNC]  Async race conditions, missing await, gather  return_exceptions
  [RES]    Resource leaks: files, DB connections,  '
  [MATH]    :   , mean-of-means, overflow
  [STATE]  :  , singleton abuse, mutable class attrs
  [API]      API: sklearn, pandas, torch
  [FLOW]    : unreachable code,  True/False 
  [FEAT]   Feature engineering: temporal ordering, rolling  min_periods
"""

import ast
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEVERITY = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
SEVERITY_EMOJI = {
    "CRITICAL": "[CRITICAL]", "HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]",
    "LOW": "[LOW]", "INFO": "[INFO]",
}
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".pytest_cache",
    "dist", "build",
}


@dataclass
class Issue:
    category: str
    severity: str
    file:     str
    line:     int
    message:  str
    code:     str = ""
    fix_hint: str = ""

    def __str__(self) -> str:
        emoji = SEVERITY_EMOJI.get(self.severity, "[INFO]")
        parts = [f"{emoji} [{self.severity}] [{self.category}] {self.file}:{self.line}"]
        parts.append(f"   {self.message}")
        if self.code:
            parts.append(f"   > {self.code.strip()[:120]}")
        if self.fix_hint:
            parts.append(f"   - {self.fix_hint}")
        return "\n".join(parts)


@dataclass
class LogicResult:
    issues:     list[Issue] = field(default_factory=list)
    file_count: int = 0
    line_count: int = 0
    stats:      dict[str, Any] = field(default_factory=dict)

    def add(self, *args, **kwargs) -> None:
        self.issues.append(Issue(*args, **kwargs))

    def summary(self) -> dict[str, int]:
        return dict(collections.Counter(i.severity for i in self.issues))


def iter_py_files(root: Path):
    for path in sorted(root.rglob("*.py")):
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


def get_source_line(lines: list[str], lineno: int) -> str:
    return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""


# -------------------------------------------------------------------------------
#  [LEAK]   
# -------------------------------------------------------------------------------

class LeakageChecker:
    """
       data leakage:
    1. Temporal leakage      
    2. Target leakage      train/test 
    3. Scaler leakage  fit  test/val 
    4. KFold leakage  preprocessing  split
    """

    #       train 
    FIT_METHODS = {"fit", "fit_transform"}

    #      test/val 
    TEST_VAR_PATTERNS = re.compile(
        r'\b(:X_test|x_test|X_val|x_val|test_df|val_df|df_test|df_val)\b'
    )

    #  temporal leakage
    #  Exclude validation tools that perform intentional look-ahead checks
    SAFE_FILES = {
        "backtesting/advanced/advanced_engine.py", 
        "validation/data_leakage_detector.py", 
        "models/analysis/overfitting_detector.py", 
        "optimization/hyperparameters/bayesian.py", 
        "pipeline/stages/stage_0_data_generation.py", 
        "data/synthetic/data_generator.py",
        "validation/time_series_validator.py",
        "features/validation/feature_leakage_guard.py",
        "pipeline/guards/temporal_leakage_guard.py",
        "pipeline/guards/temporal_target_guard.py",
        "risk/elite_risk_metrics.py"
    }

    FUTURE_DATA_PATTERNS = [
        (re.compile(r'shift\s*\(\s*-\s*[1-9]'),
         "shift(-N)     (temporal leakage)"),
        (re.compile(r'\.rolling\([^)]+\)(!\.shift)'),
         "rolling()  .shift(1)       (look-ahead bias)"),
        (re.compile(r'expanding\(\)(!\.shift)'),
         "expanding()  .shift(1)   look-ahead bias"),
    ]

    # Target   features
    TARGET_IN_FEATURES = re.compile(r'target_\w+|y_train|y_test')

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        #   AST  fit()  test- 
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # scaler.fit(X_test)  scaler.fit_transform(X_test)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.FIT_METHODS:
                        for arg in node.args:
                            arg_src = ast.unparse(arg) if hasattr(ast, 'unparse') else ""
                            if self.TEST_VAR_PATTERNS.search(arg_src):
                                result.add(
                                    "LEAK", "CRITICAL", fname, node.lineno,
                                    f"Scaler/model.{node.func.attr}()  test/val   data leakage",
                                    get_source_line(lines, node.lineno),
                                    fix_hint="fit()   train,  transform()  test/val "
                                )

        #    
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or "# audit-ignore" in line:
                continue

            # Temporal leakage 
            for pattern, msg in self.FUTURE_DATA_PATTERNS:
                if pattern.search(line):
                    # Check if file is in the safe list (normalize for cross-platform matching)
                    normalized_fname = fname.replace("\\", "/")
                    if any(safe in normalized_fname for safe in self.SAFE_FILES):
                        continue
                    
                    # Check for explicit .shift(1) usage
                    context = " ".join(lines[max(0, i-2):i+2])
                    if "rolling" in msg and ".shift(1)" in context:
                        continue  # OK   shift
                    result.add(
                        "LEAK", "HIGH", fname, i, msg, stripped,
                        fix_hint=" .shift(1)  rolling/expanding   look-ahead bias"
                    )

            # Train-test concatenation  fit
            if re.search(r'pd\.concat\s*\(\s*\[.*(:train|test).*(:test|train)', line):
                result.add(
                    "LEAK", "HIGH", fname, i,
                    "concat train+test   data leakage    fit()",
                    stripped,
                    fix_hint="  preprocessing (fit)   concat    exploratory"
                )

            # Target columns  feature matrix
            if re.search(r'features\[.*[\'"]target_', line) or re.search(r'X\s*=.*[\'"]target_', line):
                result.add(
                    "LEAK", "CRITICAL", fname, i,
                    "Target   feature matrix  target leakage",
                    stripped,
                    fix_hint=" target_ : X = df[[c for c in df.columns if not c.startswith('target_')]]"
                )

            # iloc/loc  reset_index  merge/concat
            if re.search(r'\.merge\(|pd\.concat', line):
                next_lines = " ".join(lines[i:min(i+5, len(lines))])
                if "reset_index" not in next_lines and "iloc" in next_lines:
                    result.add(
                        "LEAK", "MEDIUM", fname, i,
                        "merge/concat  reset_index  .iloc[]     ",
                        stripped,
                        fix_hint=" .reset_index(drop=True)  merge/concat"
                    )


# -------------------------------------------------------------------------------
#  [PANDAS]  - pandas
# -------------------------------------------------------------------------------

class PandasChecker:
    """ pandas -     ."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for i, line in enumerate(lines, 1):
            if "# audit-ignore" in line:
                continue
            # ... rest of the logic


# -------------------------------------------------------------------------------
#  [MATH]   
# -------------------------------------------------------------------------------

class MathChecker:
    """    ."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):

            #        
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                divisor_src = ast.unparse(node.right) if hasattr(ast, 'unparse') else ""

                #    
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    result.add(
                        "MATH", "CRITICAL", fname, node.lineno,
                        "    ZeroDivisionError",
                        get_source_line(lines, node.lineno),
                        fix_hint="    np.divide(a, b, where=b!=0)"
                    )

                # len()    
                elif "len(" in divisor_src:
                    context = "\n".join(lines[max(0, node.lineno-5):node.lineno])
                    if "if len" not in context and "assert len" not in context:
                        result.add(
                            "MATH", "MEDIUM", fname, node.lineno,
                            f"  len()    len() > 0",
                            get_source_line(lines, node.lineno),
                            fix_hint="if n > 0: result = total / n else: result = 0"
                        )

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Mean of means   
            if re.search(r'\.mean\(\s*\).*\.mean\(\s*\)', line):
                result.add(
                    "MATH", "HIGH", fname, i,
                    "Mean of means       ",
                    stripped,
                    fix_hint=" : (sum * count).sum() / count.sum()"
                )

            # np.log    0
            if re.search(r'np\.log\s*\((!1\s*\+)', line):
                context = " ".join(lines[max(0,i-3):i+1])
                if "clip" not in context and "where" not in context and "maximum" not in context:
                    result.add(
                        "MATH", "HIGH", fname, i,
                        "np.log()    0  '   -inf/NaN",
                        stripped,
                        fix_hint="np.log(np.maximum(x, 1e-8))  np.log1p(x)   "
                    )

            #  float  ==
            if re.search(r'==\s*0\.0|==\s*1\.0|0\.0\s*==|1\.0\s*==', line):
                result.add(
                    "MATH", "MEDIUM", fname, i,
                    "  float  ==    floating point precision",
                    stripped,
                    fix_hint="np.isclose(a, 0.0)  abs(a - 0.0) < 1e-9"
                )

            #  sharpe ratio
            if re.search(r'sharpe', line, re.IGNORECASE):
                if re.search(r'mean\s*\(\s*\)\s*/\s*std\s*\(\s*\)', line):
                    result.add(
                        "MATH", "MEDIUM", fname, i,
                        "Sharpe ratio   risk-free rate  annualization",
                        stripped,
                        fix_hint="sharpe = (returns.mean() - rf_rate) / returns.std() * sqrt(252)"
                    )

            # Percentage change  +1  denominator
            if re.search(r'pct_change|percentage.change', line, re.IGNORECASE):
                context = " ".join(lines[max(0,i-2):i+3])
                if "fill" not in context and "na" not in context.lower():
                    result.add(
                        "MATH", "LOW", fname, i,
                        "pct_change()   NaN   ",
                        stripped,
                        fix_hint="df.pct_change().fillna(0)  .dropna()"
                    )

            # Integer division   float
            if re.search(r'//\s*\d+', line) and re.search(r'(:ratio|rate|weight|prob)', line, re.IGNORECASE):
                result.add(
                    "MATH", "MEDIUM", fname, i,
                    "Integer division //  ratio/rate/weight    ",
                    stripped,
                    fix_hint=" //  /   "
                )


# -------------------------------------------------------------------------------
#  [LOGIC]   
# -------------------------------------------------------------------------------

class LogicChecker:
    """  : unreachable code, , ."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):

            # Unreachable code  return/raise/continue/break
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_unreachable(node, fname, lines, result)

            # : if x == x, if True, if not False
            if isinstance(node, ast.If):
                self._check_tautology(node, fname, lines, result)

            #  except    
            if isinstance(node, ast.ExceptHandler):
                if node.type is None and len(node.body) == 1:
                    if isinstance(node.body[0], ast.Pass):
                        result.add(
                            "LOGIC", "HIGH", fname, node.lineno,
                            "except: pass     ",
                            get_source_line(lines, node.lineno),
                            fix_hint="   logger.warning(exc)  raise"
                        )

            # Return None   if ... return None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_redundant_none_return(node, fname, lines, result)

        #   
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # not in + not   
            if re.search(r'if\s+not\s+\w+\s+not\s+in', line):
                result.add(
                    "LOGIC", "MEDIUM", fname, i,
                    "  `not x not in`     ",
                    stripped,
                    fix_hint="if x in collection: ( not not)"
                )

            #  bool  is  ==
            if re.search(r'is\s+True|is\s+False', line) and "isinstance" not in line:
                result.add(
                    "LOGIC", "LOW", fname, i,
                    "`is True` / `is False`   `== True`   `if x:`",
                    stripped,
                    fix_hint="if x:  if x is True:"
                )

            # or=  |=  
            if re.search(r'\w+\s*=\s*\w+\s*or\s*\w+', line):
                if re.search(r'set\(|{.*}', " ".join(lines[max(0,i-5):i])):
                    result.add(
                        "LOGIC", "LOW", fname, i,
                        "    `|=`  `or=`  union ",
                        stripped
                    )

            # Status check   
            if re.search(r'\.get\s*\(\s*["\']status["\']\s*\)\s*==\s*["\']', line):
                result.add(
                    "LOGIC", "LOW", fname, i,
                    ".get('status')  default   None   ",
                    stripped,
                    fix_hint=".get('status', 'unknown') == 'success'"
                )

    def _check_unreachable(self, func_node, fname, lines, result):
        """   return/raise/continue."""
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func_node:
                continue

        #     return    
        if isinstance(func_node.body[-1], ast.Return):
            return  # OK  return  

        #  return        
        for i, stmt in enumerate(func_node.body[:-1]):
            if isinstance(stmt, ast.Return):
                next_stmt = func_node.body[i + 1]
                if not isinstance(next_stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                              ast.ClassDef)):
                    result.add(
                        "LOGIC", "MEDIUM", fname, next_stmt.lineno,
                        f"Unreachable code  return  {func_node.name}()",
                        get_source_line(lines, next_stmt.lineno),
                        fix_hint="     return  "
                    )

    def _check_tautology(self, node, fname, lines, result):
        """ True   False ."""
        # if True:  if False:
        if isinstance(node.test, ast.Constant):
            if node.test.value is True:
                result.add(
                    "LOGIC", "MEDIUM", fname, node.lineno,
                    "if True:     (  )",
                    get_source_line(lines, node.lineno),
                    fix_hint="      "
                )
            elif node.test.value is False:
                result.add(
                    "LOGIC", "HIGH", fname, node.lineno,
                    "if False:      (dead code)",
                    get_source_line(lines, node.lineno),
                    fix_hint="    "
                )

        # x == x 
        if isinstance(node.test, ast.Compare):
            if (len(node.test.ops) == 1 and
                    isinstance(node.test.ops[0], ast.Eq)):
                left_src  = ast.unparse(node.test.left) if hasattr(ast, 'unparse') else ""
                right_src = ast.unparse(node.test.comparators[0]) if hasattr(ast, 'unparse') else ""
                if left_src and left_src == right_src:
                    result.add(
                        "LOGIC", "HIGH", fname, node.lineno,
                        f": `{left_src} == {right_src}`  True",
                        get_source_line(lines, node.lineno)
                    )

    def _check_redundant_none_return(self, func_node, fname, lines, result):
        """if x: return None + else     None."""
        pass  #       


# -------------------------------------------------------------------------------
#  [ASYNC]  async race conditions  
# -------------------------------------------------------------------------------

class AsyncChecker:
    """   async ."""

    @staticmethod
    def _sync_function_has_direct_await(func_node: ast.FunctionDef) -> bool:
        class DirectAwaitVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_await = False

            def visit_Await(self, node):
                self.has_await = True

            def visit_FunctionDef(self, node):
                return

            def visit_AsyncFunctionDef(self, node):
                return

            def visit_ClassDef(self, node):
                return

        visitor = DirectAwaitVisitor()
        for stmt in func_node.body:
            visitor.visit(stmt)
            if visitor.has_await:
                return True
        return False

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):

            # asyncio.gather  return_exceptions=True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and
                            node.func.value.id == "asyncio" and
                            node.func.attr == "gather"):
                        has_return_exc = any(
                            isinstance(kw, ast.keyword) and
                            kw.arg == "return_exceptions"
                            for kw in node.keywords
                        )
                        if not has_return_exc:
                            result.add(
                                "ASYNC", "HIGH", fname, node.lineno,
                                "asyncio.gather()  return_exceptions=True      ",
                                get_source_line(lines, node.lineno),
                                fix_hint="asyncio.gather(*tasks, return_exceptions=True)"
                            )

            # Await    ( async)
            if isinstance(node, ast.FunctionDef) and self._sync_function_has_direct_await(node):
                result.add(
                    "ASYNC", "CRITICAL", fname, node.lineno,
                    f"await inside sync function {node.name}()",
                    get_source_line(lines, node.lineno),
                    fix_hint=f"async def {node.name}(...)"
                )

            #  Task   reference
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("create_task", "ensure_future"):
                        #    
                        parent_assign = False
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.Assign):
                                if any(
                                    isinstance(v, ast.Call) and v is node
                                    for v in ast.walk(parent.value)
                                ):
                                    parent_assign = True
                                    break
                        if not parent_assign:
                            result.add(
                                "ASYNC", "MEDIUM", fname, node.lineno,
                                f"create_task/ensure_future   reference  task   garbage collected",
                                get_source_line(lines, node.lineno),
                                fix_hint="task = asyncio.create_task(coro())    "
                            )

        #  
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # time.sleep()  async 
            if re.search(r'time\.sleep\s*\(', line):
                context = "\n".join(lines[max(0,i-10):i])
                if "async def" in context:
                    result.add(
                        "ASYNC", "HIGH", fname, i,
                        "time.sleep()  async    event loop",
                        stripped,
                        fix_hint="await asyncio.sleep(seconds)"
                    )

            # requests   async 
            if re.search(r'requests\.(:get|post|put|delete)\s*\(', line):
                context = "\n".join(lines[max(0,i-15):i])
                if "async def" in context:
                    result.add(
                        "ASYNC", "HIGH", fname, i,
                        "requests.get() ()  async    event loop",
                        stripped,
                        fix_hint=" aiohttp.ClientSession()  httpx.AsyncClient()"
                    )

            # Mutex/Lock  async with
            if re.search(r'threading\.Lock\(\)|asyncio\.Lock\(\)', line):
                context = "\n".join(lines[i:min(i+5, len(lines))])
                if "with" not in context and "acquire" not in context:
                    result.add(
                        "ASYNC", "MEDIUM", fname, i,
                        "Lock()      deadlock  ",
                        stripped,
                        fix_hint="async with lock: ...  with lock: ..."
                    )


# -------------------------------------------------------------------------------
#  [RES]  resource leaks
# -------------------------------------------------------------------------------

class ResourceChecker:
    """  ."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for node in ast.walk(tree):
            # open()  with
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    #     `with` statement
                    is_in_with = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.With):
                            for item in parent.items:
                                if item.context_expr is node:
                                    is_in_with = True
                    if not is_in_with:
                        #    .close()   AST
                        result.add(
                            "RES", "HIGH", fname, node.lineno,
                            "open()  `with`       ",
                            get_source_line(lines, node.lineno),
                            fix_hint="with open(path) as f: ..."
                        )

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # DB connection  close  without context manager
            if re.search(r'duckdb\.connect\s*\(|sqlite3\.connect\s*\(', line):
                context = " ".join(lines[i:min(i+20, len(lines))])
                if ".close()" not in context and "with " not in " ".join(lines[max(0,i-2):i]):
                    result.add(
                        "RES", "MEDIUM", fname, i,
                        "DB connection   close()  context manager",
                        stripped,
                        fix_hint="with duckdb.connect(path) as con: ...  con.close()  finally"
                    )

            # Thread  daemon=True  join()
            if re.search(r'threading\.Thread\s*\(', line):
                context = " ".join(lines[i:min(i+5, len(lines))])
                if "daemon=True" not in line and ".join()" not in context and ".start()" in context:
                    result.add(
                        "RES", "LOW", fname, i,
                        "threading.Thread  daemon=True  .join()     ",
                        stripped,
                        fix_hint="daemon=True  background threads  t.join()  "
                    )

            # HTTP session  close
            if re.search(r'requests\.Session\s*\(\s*\)', line):
                context = " ".join(lines[i:min(i+30, len(lines))])
                if ".close()" not in context and "with " not in " ".join(lines[max(0,i-2):i]):
                    result.add(
                        "RES", "MEDIUM", fname, i,
                        "requests.Session()  close()  context manager",
                        stripped,
                        fix_hint="with requests.Session() as session: ..."
                    )


# -------------------------------------------------------------------------------
#  [FEAT]  feature engineering 
# -------------------------------------------------------------------------------

class FeatureEngineeringChecker:
    """  feature engineering  ML trading pipeline."""

    TEMPORAL_NAME_SAFE_FILES = {
        "algorithms/bias_detector.py",
        "features/validation/feature_leakage_guard.py",
        "pipeline/guards/temporal_leakage_guard.py",
    }

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or "# audit-ignore" in line:
                continue

            # rolling  min_periods
            if re.search(r'\.rolling\s*\(\s*(:window\s*=\s*)\d+\s*\)', line):
                if "min_periods" not in line:
                    result.add(
                        "FEAT", "MEDIUM", fname, i,
                        "rolling()  min_periods   N   NaN    features",
                        stripped,
                        fix_hint="df.rolling(window=N, min_periods=1).mean()"
                    )

            # ewm  adjust=False  online learning
            if re.search(r'\.ewm\s*\(', line):
                if "adjust" not in line:
                    result.add(
                        "FEAT", "LOW", fname, i,
                        "ewm()   adjust=      ",
                        stripped,
                        fix_hint="ewm(span=N, adjust=False)   EMA"
                    )

            # RSI   0-100
            if re.search(r'\brsi\b|\bRSI\b', line):
                context = " ".join(lines[max(0,i-3):i+3])
                # Skip if it looks like a comment, docstring, import, or pure config/assignment
                if (stripped.startswith("#") or 
                    re.search(r'import|def|class|version|config|description|key|column', stripped, re.IGNORECASE) or
                    re.search(r'clip|clamp|min.*max|0.*100', line)):
                    continue
                result.add(
                    "FEAT", "LOW", fname, i,
                    "RSI   [0, 100]     ",
                    stripped,
                    fix_hint="rsi = rsi.clip(0, 100)"
                )

            #     std
            if re.search(r'/\s*(:std|std\(\))', line):
                context = " ".join(lines[max(0,i-5):i+1])
                if "replace" not in context and "clip" not in context and "where" not in context:
                    result.add(
                        "FEAT", "HIGH", fname, i,
                        "  std()    std=0   NaN/inf   ",
                        stripped,
                        fix_hint="std = std.replace(0, 1)  / (std + 1e-8)"
                    )

            # Temporal feature   
            if re.search(r'future_return|next_price|tomorrow', line, re.IGNORECASE):
                normalized_fname = fname.replace("\\", "/")
                if any(safe in normalized_fname for safe in self.TEMPORAL_NAME_SAFE_FILES):
                    continue
                if not re.search(r'target|label|y_', line, re.IGNORECASE):
                    result.add(
                        "FEAT", "HIGH", fname, i,
                        "  'future/next/tomorrow'  features   target leakage",
                        stripped,
                        fix_hint="   ,   feature: prefix 'target_'"
                    )

            #    split
            if re.search(r'MinMaxScaler|StandardScaler|RobustScaler', line):
                if re.search(r'fit_transform', line):
                    context = " ".join(lines[max(0,i-20):i])
                    if "train_test_split" not in context and "X_train" not in line:
                        result.add(
                            "FEAT", "CRITICAL", fname, i,
                            "Scaler.fit_transform()  train_test_split  scaler  test  (leakage)",
                            stripped,
                            fix_hint=" train_test_split,  scaler.fit(X_train), scaler.transform(X_test)"
                        )

            # Infinite values  log/div
            if re.search(r'np\.log|np\.sqrt', line):
                context = " ".join(lines[max(0,i-3):i+3])
                if "isinf" not in context and "isnan" not in context and "replace" not in context:
                    result.add(
                        "FEAT", "MEDIUM", fname, i,
                        f"np.log/sqrt   inf/NaN  ",
                        stripped,
                        fix_hint="df.replace([np.inf, -np.inf], np.nan).fillna(0)"
                    )


# -------------------------------------------------------------------------------
#  [STATE]     singleton 
# -------------------------------------------------------------------------------

class StateChecker:
    """   :  , mutable class attrs."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        # Mutable class attributes (  __init__)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set)):
                                    result.add(
                                        "STATE", "HIGH", fname, stmt.lineno,
                                        f"Mutable class attribute {node.name}.{target.id} = []  "
                                        f"    ",
                                        get_source_line(lines, stmt.lineno),
                                        fix_hint=f"def __init__(self): self.{target.id} = []"
                                    )

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # global keyword
            if re.search(r'^\s*global\s+\w+', line):
                result.add(
                    "STATE", "MEDIUM", fname, i,
                    "global      ",
                    stripped,
                    fix_hint="       "
                )

            # Singleton   
            if re.search(r'^_\w+:\s*\w+\s*\|\s*None\s*=\s*None', line):
                result.add(
                    "STATE", "LOW", fname, i,
                    " singleton       concurrent ",
                    stripped,
                    fix_hint=" threading.Lock()  thread-safe singleton"
                )


# -------------------------------------------------------------------------------
#  [API]     API
# -------------------------------------------------------------------------------

class APIUsageChecker:
    """   sklearn, pandas, numpy API."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: LogicResult) -> None:
        fname = rel(path, root)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # train_test_split  stratify  
            if re.search(r'train_test_split\s*\(', line):
                if "stratify" not in line and "random_state" not in line:
                    result.add(
                        "API", "MEDIUM", fname, i,
                        "train_test_split  random_state   ",
                        stripped,
                        fix_hint="train_test_split(X, y, test_size=0.2, random_state=42)"
                    )

            # cross_val_score  shuffle
            if re.search(r'cross_val_score|KFold\s*\(', line):
                if "shuffle" not in line and "TimeSeriesSplit" not in line:
                    result.add(
                        "API", "HIGH", fname, i,
                        "KFold/cross_val_score  shuffle=True  time series  "
                        "  ,  TimeSeriesSplit",
                        stripped,
                        fix_hint="TimeSeriesSplit(n_splits=5)  temporally ordered data"
                    )

            # predict_proba  predict   
            if re.search(r'\.predict\s*\(\s*X_test\s*\)', line):
                context = " ".join(lines[max(0,i-20):i])
                if re.search(r'accuracy_score|roc_auc', context):
                    result.add(
                        "API", "LOW", fname, i,
                        "predict()  ROC AUC   predict_proba()[:,1]",
                        stripped,
                        fix_hint="roc_auc_score(y_test, model.predict_proba(X_test)[:,1])"
                    )

            # LightGBM/XGBoost: verbose=-1 
            if re.search(r'LGBMClassifier|LGBMRegressor|XGBClassifier|XGBRegressor', line):
                if "verbose" not in line and "silent" not in line:
                    result.add(
                        "API", "INFO", fname, i,
                        "LightGBM/XGBoost  verbose=-1     ",
                        stripped,
                        fix_hint="LGBMClassifier(verbose=-1)  XGBClassifier(verbosity=0)"
                    )

            # numpy random  seed
            if re.search(r'np\.random\.(:rand|randn|randint|choice)\s*\(', line):
                context = " ".join(lines[max(0,i-30):i])
                if "np.random.seed" not in context and "rng" not in context:
                    result.add(
                        "API", "LOW", fname, i,
                        "np.random  seed   ",
                        stripped,
                        fix_hint="rng = np.random.default_rng(42); rng.random(...)"
                    )

            #    
            if re.search(r'\+\s*=\s*["\']|["\'].*\+.*str\(', line):
                context = " ".join(lines[max(0,i-3):i+1])
                if "for " in context or "while " in context:
                    result.add(
                        "API", "LOW", fname, i,
                        "     O(n)   list + join()",
                        stripped,
                        fix_hint="parts = []; parts.append(s); result = ''.join(parts)"
                    )


# -------------------------------------------------------------------------------
#   
# -------------------------------------------------------------------------------

class DeepLogicAuditor:

    def __init__(self, root: Path) -> None:
        self.root    = root
        self.result  = LogicResult()
        self.checkers = [
            LeakageChecker(),
            PandasChecker(),
            MathChecker(),
            LogicChecker(),
            AsyncChecker(),
            ResourceChecker(),
            FeatureEngineeringChecker(),
            StateChecker(),
            APIUsageChecker(),
        ]

    def audit(self) -> LogicResult:
        print(f" Deep Logic Audit: {self.root}")
        files = list(iter_py_files(self.root))
        self.result.file_count = len(files)

        for path in files:
            text, lines = read_file(path)
            self.result.line_count += len(lines)
            if not text:
                continue

            tree = parse_ast(text)
            if tree is None:
                continue

            for checker in self.checkers:
                try:
                    checker.run(path, self.root, text, lines, tree, self.result)
                except Exception as exc:
                    print(f"   Checker {checker.__class__.__name__} failed on {path}: {exc}")

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


def print_report(result: LogicResult, show_fix: bool = True, max_issues: int = 1000) -> None:
    sev_order = list(SEVERITY.keys())
    issues_sorted = sorted(
        result.issues,
        key=lambda i: (SEVERITY.get(i.severity, 99), i.category, i.file, i.line),
    )

    current_sev = None
    shown = 0
    for issue in issues_sorted:
        if shown >= max_issues:
            print(f"\n...   {len(issues_sorted) - shown} issues")
            break
        if issue.severity != current_sev:
            current_sev = issue.severity
            print(f"\n{'-' * 70}")
            print(f"  {SEVERITY_EMOJI[issue.severity]} {issue.severity}")
            print(f"{'-' * 70}")
        print(f"\n{SEVERITY_EMOJI[issue.severity]} [{issue.category}] {issue.file}:{issue.line}")
        print(f"   {issue.message}")
        if issue.code:
            print(f"   > {issue.code[:110]}")
        if show_fix and issue.fix_hint:
            print(f"   - {issue.fix_hint}")
        shown += 1

    s = result.stats
    print(f"\n{'-' * 70}")
    print(f"   DEEP LOGIC AUDIT  ")
    print(f"{'-' * 70}")
    print(f"   : {s['files']}   : {s['lines']:,}   Issues: {s['issues']}")
    print()
    for sev in sev_order:
        cnt = s['by_severity'].get(sev, 0)
        if cnt:
            print(f"  {SEVERITY_EMOJI[sev]} {sev:<12}: {cnt}")
    print()
    print("   :")
    for cat, cnt in sorted(s['by_category'].items(), key=lambda x: -x[1]):
        desc = {
            "LEAK":   " ",
            "PANDAS": "Pandas -",
            "MATH":   " ",
            "LOGIC":  " ",
            "ASYNC":  "Async ",
            "RES":    "Resource leaks",
            "FEAT":   "Feature engineering",
            "STATE":  "/Singleton",
            "API":    "API ",
        }.get(cat, "")
        print(f"    [{cat}] {cnt:3}  {desc}")
    print(f"{'-' * 70}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DEAN Deep Logic Auditor")
    parser.add_argument("--root",       default=".",    help=" ")
    parser.add_argument("--json",       action="store_true")
    parser.add_argument("--output",     default="",     help="  ")
    parser.add_argument("--severity",   default="LOW",  help="CRITICAL/HIGH/MEDIUM/LOW/INFO")
    parser.add_argument("--category",   default="",     help="LEAK,PANDAS,MATH,LOGIC,ASYNC,RES,FEAT,STATE,API")
    parser.add_argument("--max-issues", default=1000,   type=int)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR:  : {root}")
        sys.exit(1)

    auditor = DeepLogicAuditor(root)
    result  = auditor.audit()

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
                    {"category": i.category, "severity": i.severity,
                     "file": i.file, "line": i.line,
                     "message": i.message, "fix_hint": i.fix_hint}
                    for i in result.issues
                ],
            },
            ensure_ascii=False, indent=2,
        )
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"DONE: {args.output}")
        else:
            print(output)
    else:
        if args.output:
            import contextlib
            with open(args.output, "w", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f):
                    print_report(result, True, args.max_issues)
            print(f"DONE: {args.output}")
        else:
            print_report(result, True, args.max_issues)

    critical = result.stats.get("by_severity", {}).get("CRITICAL", 0)
    high     = result.stats.get("by_severity", {}).get("HIGH", 0)
    sys.exit(2 if critical > 0 else (1 if high > 0 else 0))


if __name__ == "__main__":
    main()
