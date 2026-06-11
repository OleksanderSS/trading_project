from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext


class AgentRegistry:
    def __init__(self, config_path: str | Path, project_root: str | Path | None = None):
        self.config_path = Path(config_path)
        self.project_root = Path(project_root or self.config_path.parents[2]).resolve()
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return raw.get("agents", {})

    def load_all(self, context: MarketContext | None = None) -> list[BaseAgent]:
        context = context or MarketContext()
        agents: list[BaseAgent] = []
        for name, cfg in self._config.items():
            if not cfg.get("enabled", False):
                continue
            agent_cfg = dict(cfg)
            agent_cfg["project_root"] = str(self.project_root)
            agent = self._instantiate(name, agent_cfg)
            if agent.check_prerequisites(context):
                agents.append(agent)
        return agents

    def load_branch(self, branch: str, context: MarketContext | None = None) -> list[BaseAgent]:
        return [agent for agent in self.load_all(context) if agent.config.get("branch") == branch]

    def _instantiate(self, name: str, cfg: dict[str, Any]) -> BaseAgent:
        module_name, class_name = cfg["class_path"].split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        agent_cls = getattr(module, class_name)
        return agent_cls(name=name, config=cfg)
