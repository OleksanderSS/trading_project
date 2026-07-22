from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar

import yaml

from dean_os.base import BaseAgent
from dean_os.schemas import EvidenceItem, MarketContext, PipelineReport


class AgentRegistry:
    # Allowed module prefixes for agent class paths
    ALLOWED_PREFIXES: ClassVar[list[str]] = [
        "dean_os",
    ]

    def __init__(
        self,
        config_path: str | Path,
        project_root: str | Path | None = None,
        overrides: dict[str, dict[str, Any]] | None = None,
    ):
        self.config_path = Path(config_path)
        self.project_root = Path(project_root or self.config_path.parents[2]).resolve()
        self.overrides = overrides or {}
        self._config = self._load_config()
        self._load_errors: dict[str, str] = {}
        self._synthetic_reports: dict[str, PipelineReport] = {}

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        agents = raw.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("Registry 'agents' must be a mapping")
        merged = {name: dict(cfg or {}) for name, cfg in agents.items()}
        for name, override in self.overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"Registry override for {name!r} must be a mapping")
            merged.setdefault(name, {}).update(override)
        self._validate_exclusive_groups(merged)
        return merged

    def _validate_exclusive_groups(
        self,
        agents: dict[str, Any],
    ) -> None:
        enabled_groups: dict[str, list[tuple[str, set[str] | None]]] = {}
        for name, cfg in agents.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", False):
                continue
            group = str(cfg.get("execution_group") or "").strip()
            if not group:
                continue
            raw_phases = cfg.get("run_phases")
            phases = (
                {
                    str(phase).strip()
                    for phase in raw_phases
                    if str(phase).strip()
                }
                if raw_phases
                else None
            )
            enabled_groups.setdefault(group, []).append(
                (name, phases)
            )
        for group, members in enabled_groups.items():
            for index, (left_name, left_phases) in enumerate(members):
                for right_name, right_phases in members[index + 1 :]:
                    overlaps = (
                        left_phases is None
                        or right_phases is None
                        or bool(left_phases & right_phases)
                    )
                    if overlaps:
                        raise ValueError(
                            "Enabled agents share exclusive execution "
                            f"group {group!r} in overlapping phases: "
                            f"{left_name}, {right_name}"
                        )

    def _validate_class_path(self, class_path: str) -> None:
        """
        Validate that a class path is within allowed prefixes.

        Args:
            class_path: Class path in format "module:ClassName"

        Raises:
            ValueError: If class path is malformed or outside allowed prefixes
        """
        if ":" not in class_path:
            raise ValueError(f"Malformed class_path: {class_path}. Expected format: 'module:ClassName'")

        module_name, class_name = class_path.split(":", maxsplit=1)

        if not module_name or not class_name:
            raise ValueError(f"Malformed class_path: {class_path}. Module and class name must be non-empty")

        # Check if module is in allowed prefixes
        is_allowed = any(
            module_name == prefix
            or module_name.startswith(prefix + ".")
            for prefix in self.ALLOWED_PREFIXES
        )
        if not is_allowed:
            raise ValueError(
                f"Class path module '{module_name}' is not in allowed prefixes: {self.ALLOWED_PREFIXES}"
            )

    def _is_hard_block_agent(self, cfg: dict[str, Any]) -> bool:
        """Check if an agent is configured as hard/block."""
        veto_level = cfg.get("veto_level", "none")
        error_behavior = cfg.get("error_behavior", "skip")
        return veto_level == "hard" and error_behavior == "block"

    def _create_synthetic_blocked_report(self, name: str, cfg: dict[str, Any], reason: str) -> PipelineReport:
        """Create a synthetic blocked PipelineReport for agents with missing prerequisites."""
        return PipelineReport(
            agent_name=name,
            agent_version=cfg.get("version", "unknown"),
            verdict="blocked",
            confidence=1.0,
            data_quality_score=0.0,
            signal_strength=-1.0,
            reasons=[reason],
            risks=[f"Hard/block agent {name} cannot function: {reason}"],
            blind_spots=["Agent not loaded due to missing prerequisites"],
            evidence=[
                EvidenceItem(
                    source_type="audit_finding",
                    source="agent_registry",
                    key=f"{name}_missing_prerequisite",
                    value=reason,
                )
            ],
            input_hash="",
            metrics_snapshot={
                "synthetic": True,
                "missing_prerequisites": True,
                "veto_level": cfg.get("veto_level", "none"),
                "error_behavior": cfg.get("error_behavior", "skip"),
            },
        )

    def load_all(self, context: MarketContext | None = None) -> list[BaseAgent]:
        context = context or MarketContext()
        self._load_errors = {}
        self._synthetic_reports = {}
        agents: list[BaseAgent] = []
        for name, cfg in self._config.items():
            if not cfg.get("enabled", False):
                continue
            run_phases = cfg.get("run_phases")
            if run_phases and context.phase not in {
                str(phase).strip()
                for phase in run_phases
                if str(phase).strip()
            }:
                continue

            try:
                # Validate class path before instantiation
                class_path = cfg.get("class_path")
                if not class_path:
                    self._load_errors[name] = "Missing class_path in configuration"
                    if self._is_hard_block_agent(cfg):
                        self._synthetic_reports[name] = self._create_synthetic_blocked_report(
                            name, cfg, "Missing class_path in configuration"
                        )
                    continue

                self._validate_class_path(class_path)

                agent_cfg = dict(cfg)
                agent_cfg["project_root"] = str(self.project_root)

                # Check prerequisites before instantiation
                # For hard/block agents, we need to instantiate to check prerequisites
                # but if prerequisites fail, we create a synthetic blocked report
                agent = self._instantiate(name, agent_cfg)

                # Validate that the agent is a BaseAgent subclass
                if not isinstance(agent, BaseAgent):
                    self._load_errors[name] = f"Agent class {class_path} is not a BaseAgent subclass"
                    if self._is_hard_block_agent(cfg):
                        self._synthetic_reports[name] = self._create_synthetic_blocked_report(
                            name, cfg, "Agent class is not a BaseAgent subclass"
                        )
                    continue
                if not agent.should_run_in_phase(context):
                    continue

                if not agent.check_prerequisites(context):
                    self._load_errors[name] = "Prerequisites check failed"
                    if self._is_hard_block_agent(cfg):
                        required_inputs = cfg.get("required_inputs", [])
                        reason = f"Missing required inputs: {required_inputs}" if required_inputs else "Prerequisites check failed"
                        self._synthetic_reports[name] = self._create_synthetic_blocked_report(name, cfg, reason)
                    continue

                agents.append(agent)

            except ValueError as e:
                self._load_errors[name] = f"Configuration error: {e}"
                if self._is_hard_block_agent(cfg):
                    self._synthetic_reports[name] = self._create_synthetic_blocked_report(name, cfg, str(e))
            except Exception as e:
                self._load_errors[name] = f"Instantiation error: {e}"
                if self._is_hard_block_agent(cfg):
                    self._synthetic_reports[name] = self._create_synthetic_blocked_report(name, cfg, str(e))

        return agents

    def load_branch(self, branch: str, context: MarketContext | None = None) -> list[BaseAgent]:
        return [agent for agent in self.load_all(context) if agent.config.get("branch") == branch]

    def _instantiate(self, name: str, cfg: dict[str, Any]) -> BaseAgent:
        module_name, class_name = cfg["class_path"].split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        agent_cls = getattr(module, class_name)

        # Validate that the class is a BaseAgent subclass
        if not issubclass(agent_cls, BaseAgent):
            raise ValueError(f"Class {class_name} is not a BaseAgent subclass")

        return agent_cls(name=name, config=cfg)

    def get_load_errors(self) -> dict[str, str]:
        """Return any errors that occurred during agent loading."""
        return self._load_errors

    def get_synthetic_reports(self) -> dict[str, PipelineReport]:
        """Return synthetic blocked reports for hard/block agents with missing prerequisites."""
        return self._synthetic_reports

    def hard_veto_agent_names(self) -> set[str]:
        return {
            name
            for name, cfg in self._config.items()
            if cfg.get("enabled", False)
            and cfg.get("veto_level") == "hard"
        }
