from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from dean_os.draft.dean_os_agent_system_v7.dean_os.causal_contracts import CausalClaimMetadata, GraphEdgeDynamics

DEPENDENCY_GRAPH_DIR = Path(__file__).resolve().parent.parent / "config" / "dependency_graphs"


class DependencyNode(BaseModel):
    id: str
    label: str
    category: str = "unknown"
    sector: str = "unknown"


class DependencyEdge(BaseModel):
    from_node: str = Field(alias="from")
    to: str
    type: str  # structural, cyclical, event_driven
    description: str = ""
    lag: str = "months"  # immediate, days, weeks, months, quarters, years
    strength: float = 0.5
    confidence: float = 0.5
    evidence: str = ""
    causal_metadata: CausalClaimMetadata = Field(
        default_factory=CausalClaimMetadata
    )
    dynamics: GraphEdgeDynamics = Field(default_factory=GraphEdgeDynamics)

    @model_validator(mode="after")
    def classify_legacy_dependency(self) -> "DependencyEdge":
        if self.dynamics == GraphEdgeDynamics():
            self.dynamics = GraphEdgeDynamics(
                strength=self.strength,
                lag_label=self.lag,
                estimate_confidence=self.confidence,
                edge_reliability=self.confidence,
                evidence_count=1 if self.evidence else 0,
                activation_state="candidate",
            )
        if self.causal_metadata != CausalClaimMetadata():
            return self
        if self.type == "structural":
            self.causal_metadata = CausalClaimMetadata(
                relation_type="economic_transmission",
                identification_method="assumed_mechanism",
                limitations=["Structural domain map; effect not identified"],
            )
        elif self.type == "cyclical":
            self.causal_metadata = CausalClaimMetadata(
                relation_type="statistical_association",
                limitations=["Cycle co-movement does not establish causality"],
            )
        else:
            self.causal_metadata = CausalClaimMetadata(
                relation_type="hypothesis_only",
                limitations=["Event-driven dependency requires causal review"],
            )
        return self


class DependencyGraph(BaseModel):
    """Structural economic dependency graph for a sector."""

    nodes: list[DependencyNode]
    edges: list[DependencyEdge]

    def get_downstream(self, node_id: str) -> list[DependencyEdge]:
        """Get edges where node_id is the source."""
        return [e for e in self.edges if e.from_node == node_id]

    def get_upstream(self, node_id: str) -> list[DependencyEdge]:
        """Get edges where node_id is the target."""
        return [e for e in self.edges if e.to == node_id]

    def find_path(
        self,
        from_node: str,
        to_node: str,
        max_depth: int = 5,
    ) -> list[list[DependencyEdge]]:
        """Find all paths between two nodes (simple BFS)."""
        paths: list[list[DependencyEdge]] = []

        def _dfs(current: str, target: str, visited: set[str], path: list[DependencyEdge]) -> None:
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(list(path))
                return
            for edge in self.edges:
                if edge.from_node == current and edge.to not in visited:
                    visited.add(edge.to)
                    path.append(edge)
                    _dfs(edge.to, target, visited, path)
                    path.pop()
                    visited.remove(edge.to)

        _dfs(from_node, to_node, {from_node}, [])
        return paths

    def traverse(
        self,
        start_nodes: list[str],
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Traverse downstream from start nodes, collecting affected nodes."""
        affected: list[dict[str, Any]] = []
        visited: set[str] = set()

        def _walk(current: str, depth: int, chain: list[str]) -> None:
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            affected.append({"node": current, "depth": depth, "chain": list(chain)})
            for edge in self.get_downstream(current):
                _walk(edge.to, depth + 1, chain + [edge.to])

        for node in start_nodes:
            _walk(node, 0, [node])

        return affected


def load_dependency_graph(domain_id: str) -> DependencyGraph | None:
    """Load dependency graph from YAML by domain_id."""
    path = DEPENDENCY_GRAPH_DIR / f"{domain_id}.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return DependencyGraph.model_validate(data)


def list_dependency_graphs() -> list[str]:
    """List available dependency graph domain IDs."""
    return sorted(p.stem for p in DEPENDENCY_GRAPH_DIR.glob("*.yaml"))


__all__ = [
    "DependencyEdge",
    "DependencyGraph",
    "DependencyNode",
    "list_dependency_graphs",
    "load_dependency_graph",
]
