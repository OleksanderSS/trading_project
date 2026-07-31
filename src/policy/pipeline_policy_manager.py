"""One place that answers "what are the bounds?" for the whole pipeline.

Risk limits, train/test split ratios and calibrated hyperparameters were
spread across five disconnected places, several of which could never be read:

RISK LIMITS
  - `src/config/risk_management.yaml` declares `max_daily_loss_pct: 0.03`,
    `max_position_size_pct: 0.10`, `max_drawdown_pct: 0.15`.
  - `PortfolioManager` reads `max_daily_loss_pct` into a field it calls
    `max_daily_drawdown_pct` -- the name mismatch that already caused one real
    kill-switch bug, where it read a key that did not exist and silently ran
    67% more permissive than configured.
  - `AdaptiveParameterManager` computes REGIME-AWARE limits
    (`max_daily_drawdown_pct` 0.06 trending-up ... 0.01 crisis) and
    `risk_per_trade_pct` 0.02 -- but only `recommendation_engine` consults it.
    The kill switch itself never did, so the regime-aware limits existed and
    were computed and then ignored by the one component that enforces.
  - `PortfolioManager.risk_per_trade_pct` defaults to 0.03 because no config
    key of that name exists anywhere, while AdaptiveParameters says 0.02.
  - `meta_learning/security/agent_permissions.py` carries its own hardcoded
    0.1 / 0.02 (and 0.05 / 0.01) tiers.

SPLIT RATIOS
  - `DEFAULT_TEST_SIZE = 0.2` in `src/training/constants.py` is what actually
    governs training.
  - `test_size: 0.2` is ALSO declared under the top-level `data_preparation`
    key (processing.yaml) and under `preparation` inside unified_config.yaml,
    and neither is read by the modeling orchestrator.
  - That orchestrator reads `config_manager.get_config('modeling')`, and
    **there is no `modeling:` key in any config file**, so `get_config`
    returns None and every `modeling_config.get(...)` in it -- `strategy`,
    `batch_size`, `max_memory_gb` and `test_size` alike -- falls through to a
    code constant. Those four settings are simply not configurable today.
  - `get_config()` resolves TOP-LEVEL YAML KEYS across the merged config, not
    file names: `get_config('processing')` is None even though
    processing.yaml exists.
  - Four more independent 0.2 defaults sit in calibration_engine.py,
    data_preparation.py, ml_analytics.py and base_neural.py.

This component does not invent a new regime model. `AdaptiveParameterManager`
already has one, fully built; the rule below composes with it rather than
duplicating it -- the standing constraint in this project is to fix or extend
the existing mechanism, never to add a parallel one.

PRECEDENCE RULE (the important part)

    Configured limits are a HARD CEILING. Regime adaptation may only TIGHTEN
    them, never loosen them.

So with `max_daily_loss_pct: 0.03` configured:
  - trending-up, where AdaptiveParameterManager wants 0.06 -> 0.03 wins
  - crisis, where it wants 0.01                            -> 0.01 wins
This resolves the 0.03-vs-0.06 conflict in the safe direction, and means
turning regime awareness on can never widen a risk budget someone set
deliberately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PipelinePolicyManager")

#: Fallback when nothing is configured anywhere. Matches
#: src/training/constants.DEFAULT_TEST_SIZE, which is what training uses today.
_FALLBACK_TEST_SIZE = 0.2
_FALLBACK_VAL_SIZE = 0.1


@dataclass(frozen=True)
class RiskLimits:
    """Resolved risk bounds for one decision context."""

    max_daily_loss_pct: float
    max_position_size_pct: float
    max_drawdown_pct: float
    risk_per_trade_pct: float
    regime: str | None = None
    #: True when regime adaptation tightened a configured ceiling.
    tightened_by_regime: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitPolicy:
    """Resolved chronological split ratios."""

    test_size: float
    val_size: float
    source: str

    @property
    def train_size(self) -> float:
        return round(1.0 - self.test_size - self.val_size, 10)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"train_size": self.train_size}


class PipelinePolicyManager:
    """Resolves pipeline policy from config, regime and (later) calibration."""

    #: Where a split ratio may legitimately be declared, most specific first.
    #:
    #: These are TOP-LEVEL YAML KEYS, not file names. `get_config()` resolves
    #: top-level keys across the merged config, so `get_config('processing')`
    #: returns None even though processing.yaml exists -- its top-level keys
    #: are `safe_fill` and `data_preparation`. An earlier version of this
    #: table used file names and would have silently resolved nothing,
    #: falling through to the builtin default while claiming to read config.
    _SPLIT_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("modeling", ("test_size",)),
        ("data_preparation", ("test_size",)),
    )

    def __init__(self, config_manager: Any, adaptive_manager: Any | None = None):
        self.config_manager = config_manager
        self._adaptive_manager = adaptive_manager
        self._risk_config = self._load_risk_config()

    # ── risk ─────────────────────────────────────────────────────────────

    def _load_risk_config(self) -> dict[str, Any]:
        strategy = self.config_manager.get_config("strategy", {}) or {}
        risk = strategy.get("risk_management", {}) or {}
        if not risk:
            logger.warning(
                "strategy.risk_management is empty; risk limits will use "
                "conservative built-in defaults."
            )
        return risk

    def _adaptive_limits(self, regime: str, asset_class: str,
                         volatility_percentile: float) -> dict[str, float]:
        """Ask AdaptiveParameterManager for regime-aware limits, if available."""
        manager = self._adaptive_manager
        if manager is None:
            return {}
        try:
            from src.trading.adaptive_parameter_manager import AssetClass, MarketRegime

            params = manager.compute_adaptive_params(
                regime=MarketRegime(str(regime).lower()),
                asset_class=AssetClass(str(asset_class).lower()),
                volatility_percentile=volatility_percentile,
            )
        except Exception as e:
            logger.warning(f"Regime-aware limits unavailable for '{regime}': {e}")
            return {}

        return {
            "max_daily_loss_pct": getattr(params, "max_daily_drawdown_pct", None),
            "max_position_size_pct": getattr(params, "max_position_size_pct", None),
            "risk_per_trade_pct": getattr(params, "risk_per_trade_pct", None),
        }

    def risk_limits(
        self,
        regime: str | None = None,
        asset_class: str = "large_cap",
        volatility_percentile: float = 0.5,
    ) -> RiskLimits:
        """Configured ceilings, tightened (never loosened) by regime."""
        cfg = self._risk_config
        ceilings = {
            "max_daily_loss_pct": float(cfg.get("max_daily_loss_pct", 0.03)),
            "max_position_size_pct": float(cfg.get("max_position_size_pct", 0.10)),
            "max_drawdown_pct": float(cfg.get("max_drawdown_pct", 0.15)),
            # No `risk_per_trade_pct` key exists in any config file; the
            # tighter of PortfolioManager's 0.03 and AdaptiveParameters' 0.02
            # is the honest default.
            "risk_per_trade_pct": float(cfg.get("risk_per_trade_pct", 0.02)),
        }

        resolved = dict(ceilings)
        tightened = False
        if regime:
            for key, adaptive_value in self._adaptive_limits(
                regime, asset_class, volatility_percentile
            ).items():
                if adaptive_value is None:
                    continue
                if float(adaptive_value) < resolved[key]:
                    resolved[key] = float(adaptive_value)
                    tightened = True

        return RiskLimits(
            **resolved, regime=regime, tightened_by_regime=tightened
        )

    # ── splits ───────────────────────────────────────────────────────────

    def split_policy(self) -> SplitPolicy:
        """Resolve the chronological split, saying where the number came from."""
        for section, path in self._SPLIT_SOURCES:
            node: Any = self.config_manager.get_config(section, {}) or {}
            for key in path:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(key)
            if isinstance(node, int | float) and 0 < float(node) < 1:
                test_size = float(node)
                val_size = self._val_size(section, default=_FALLBACK_VAL_SIZE)
                return SplitPolicy(test_size, val_size, f"{section}.{'.'.join(path)}")

        logger.info(
            "No usable test_size in config; using the built-in default "
            f"({_FALLBACK_TEST_SIZE})."
        )
        return SplitPolicy(_FALLBACK_TEST_SIZE, _FALLBACK_VAL_SIZE, "builtin_default")

    def _val_size(self, section: str, default: float) -> float:
        node = self.config_manager.get_config(section, {}) or {}
        for key in ("val_size", "validation_size", "validation_split"):
            value = node.get(key) if isinstance(node, dict) else None
            if isinstance(value, int | float) and 0 < float(value) < 1:
                return float(value)
        return default

    # ── hyperparameters ──────────────────────────────────────────────────

    def hyperparameters(self, model_type: str) -> dict[str, Any]:
        """Calibrated hyperparameters for a model type.

        Deliberately returns only what is configured today. Calibration is to
        be produced by dean_os's TuningAgent proposal lifecycle
        (pending/approved/rejected/expired, `allowed_for_production=False`
        until approved) per Agents_architecture.md section 10 -- NOT by a
        bespoke tuner bolted on here. This method is the seam that lifecycle
        will feed; until it does, it reports configured values and nothing is
        silently invented.
        """
        models_config = self.config_manager.get_config("models", {}) or {}
        per_model = models_config.get("hyperparameters", {}) or {}
        return dict(per_model.get(model_type, {}) or {})


_policy_manager: PipelinePolicyManager | None = None


def get_policy_manager(config_manager: Any = None,
                       adaptive_manager: Any = None) -> PipelinePolicyManager:
    """Process-wide policy manager."""
    global _policy_manager
    if _policy_manager is None or config_manager is not None:
        if config_manager is None:
            from src.config.unified_config_manager import get_current_config

            config_manager = get_current_config()
        if adaptive_manager is None:
            try:
                from src.trading.adaptive_parameter_manager import (
                    AdaptiveParameterManager,
                )

                adaptive_manager = AdaptiveParameterManager()
            except Exception as e:
                logger.warning(f"AdaptiveParameterManager unavailable: {e}")
        _policy_manager = PipelinePolicyManager(config_manager, adaptive_manager)
    return _policy_manager
