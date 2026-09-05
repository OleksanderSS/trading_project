"""AST scan for the arithmetic mistakes this audit kept finding.

Every pattern here comes from a defect that was actually shipped, not from a
list of things that could go wrong:

ANNUALISATION
    A hardcoded periods-per-year constant (252, 365, 12, 52...) outside the
    metrics library. This project stores 15m, 60m and 1d bars, so a
    `sqrt(252)` is right for one of them and wrong for the other two.
    DiaryEngine understated intraday Sharpe by 2.6x and 5.1x that way
    (d151cf7e).

POPULATION_STD
    `np.std(x)` with no ddof. The default is the population deviation; a
    Sharpe or Sortino wants the sample one. Same commit.

RIVAL_METRIC
    A function named after a metric the canonical library already owns
    (sharpe, sortino, calmar, drawdown, win_rate...) defined anywhere else.
    Five Sharpe implementations existed: three consolidated by an earlier
    audit, a fourth in diary_engine, a fifth in arena_battle that computed
    mean(predictions)/std(predictions) and let a constant predictor win the
    model tournament (8efca119).

SIGNED_RATIO
    Comparing against `something * <factor>` as a "better by N%" test. That
    inverts below zero: promotion used `challenger > champion * 1.15`, and
    -2.0 * 1.15 is -2.3, so a WORSE challenger cleared the bar (7c4bd621).

Import this from the contract test; it is not a test module itself.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

EXCLUDED_PARTS = {"archive", "__pycache__", "dead_pipeline_code"}

# The one module allowed to own metric formulas.
CANONICAL_METRICS = "src/metrics/financial/financial_metrics_library.py"

# Periods-per-year constants. 24 and 60 are excluded: they are hours and
# minutes far more often than they are annualisation factors.
ANNUALISATION_CONSTANTS = {252, 365, 366, 52, 12}

METRIC_NAMES = (
    "sharpe", "sortino", "calmar", "drawdown", "win_rate", "profit_factor",
)

#: Assigning a risk-free rate to a literal, anywhere but the canonical module.
#:
#: REGISTER #203, family C -- "a decision made in one place and not carried to
#: its consumers". `get_risk_free_rate` was written on 2026-08-10 precisely
#: because stage 7 published TWO Sharpe ratios for one equity curve, the gap
#: reproducing exactly as an assumed 2% against a configured 0%. Its own
#: docstring names three offenders. Two weeks later, on 2026-09-05, SIX live
#: sites still assigned 0.02 by hand: performance_attribution (twice),
#: risk_decomposition, hedge_fund_analyzer, portfolio_metrics -- the very key
#: the docstring called out -- and the Black-Litterman dataclass default.
#:
#: The fix existing is not the same as the fix arriving, and that gap is the
#: family. This is a NAMED rule for a NAMED decision rather than a general
#: family-C scanner: two general rules were measured on 2026-09-05 and both
#: missed -- the same-constant-different-value rule found 5 names, all
#: legitimately per-script, and the config-leaf-key rule found 47, almost all
#: structural keys like `type` and `description`.
RISK_FREE_NAMES = ("risk_free", "risk_free_rate", "rf_rate", "annual_rf",
                   "rf_baseline", "rf_daily", "const_rf")


@dataclass(frozen=True)
class Finding:
    kind: str
    module: str
    line: int
    context: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line}  [{self.kind}]  {self.context}"


def _is_sqrt_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sqrt"
    )


class _Scanner(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.findings: list[Finding] = []
        self._canonical = module.endswith(CANONICAL_METRICS.split("/")[-1])

    # -- annualisation constants -------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        if not self._canonical and _is_sqrt_call(node) and node.args:
            argument = node.args[0]
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, (int, float))
                and int(argument.value) in ANNUALISATION_CONSTANTS
            ):
                self.findings.append(Finding(
                    "ANNUALISATION", self.module, node.lineno,
                    f"sqrt({argument.value}) hardcodes a bar cadence",
                ))

        # np.std(x) with no ddof
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "std"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("np", "numpy")
            and not any(keyword.arg == "ddof" for keyword in node.keywords)
        ):
            self.findings.append(Finding(
                "POPULATION_STD", self.module, node.lineno,
                "np.std without ddof (population, not sample)",
            ))
        self.generic_visit(node)

    # -- a risk-free rate assigned by hand ---------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_risk_free(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_risk_free([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def _check_risk_free(self, targets, value, lineno: int) -> None:
        if self._canonical:
            return
        # `rf_baseline = 0.02 / 252` is a BinOp, not a Constant, and it was the
        # live shape in performance_attribution -- the first version of this
        # rule matched the other five sites and walked straight past the one
        # that had two defects in it at once. Unwrap the left operand.
        while isinstance(value, ast.BinOp):
            value = value.left
        if not isinstance(value, ast.Constant):
            return
        if not isinstance(value.value, float) or value.value == 0.0:
            return
        for target in targets:
            name = getattr(target, "attr", None) or getattr(target, "id", None)
            if not isinstance(name, str):
                continue
            lowered = name.lower()
            if any(token in lowered for token in RISK_FREE_NAMES):
                self.findings.append(Finding(
                    "RISK_FREE", self.module, lineno,
                    f"{name} = {value.value} instead of get_risk_free_rate()",
                ))

    # -- rival metric implementations --------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._canonical:
            lowered = node.name.lower()
            for metric in METRIC_NAMES:
                if metric in lowered and not lowered.startswith("test"):
                    self.findings.append(Finding(
                        "RIVAL_METRIC", self.module, node.lineno,
                        f"def {node.name} -- {metric} belongs to "
                        f"FinancialMetricsLibrary",
                    ))
                    break
        self.generic_visit(node)

    # -- signed relative comparisons ---------------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        for operator, comparator in zip(node.ops, node.comparators):
            if not isinstance(operator, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                continue
            if (
                isinstance(comparator, ast.BinOp)
                and isinstance(comparator.op, ast.Mult)
                and isinstance(comparator.right, ast.Constant)
                and isinstance(comparator.right.value, float)
                and comparator.right.value != 1.0
            ):
                self.findings.append(Finding(
                    "SIGNED_RATIO", self.module, node.lineno,
                    f"compared against x * {comparator.right.value} -- "
                    f"inverts if x can be negative",
                ))
        self.generic_visit(node)


def _python_files(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.rglob("*.py"))
        if not EXCLUDED_PARTS & set(path.parts)
    ]


def scan(root: Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_files(root or SOURCE_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanner = _Scanner(path.relative_to(PROJECT_ROOT).as_posix())
        scanner.visit(tree)
        findings.extend(scanner.findings)
    return findings


def by_kind(findings: list[Finding]) -> dict[str, list[Finding]]:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.kind, []).append(finding)
    return grouped


if __name__ == "__main__":
    grouped = by_kind(scan())
    for kind in sorted(grouped):
        entries = grouped[kind]
        print(f"\n=== {kind}: {len(entries)} ===")
        for entry in entries[:30]:
            print(f"  {entry}")
        if len(entries) > 30:
            print(f"  ... and {len(entries) - 30} more")
