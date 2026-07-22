"""Lightweight verification for the full-cycle to world-model boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def verify_world_model_cycle_binding(
    cycle_path: Path,
    cycle: dict[str, Any],
    world_path: Path,
    world: dict[str, Any],
) -> None:
    if cycle.get("contract") != "dean_full_system_review_cycle_v1":
        raise ValueError("unsupported full-system cycle contract")
    if (
        world.get("cycle_binding_contract")
        != "dean_full_system_cycle_world_model_bridge_v1"
    ):
        raise ValueError("world model is not a cycle-bound packet")
    binding = (
        (world.get("upstream_bindings") or {}).get("full_system_review_cycle")
        or {}
    )
    if binding.get("sha256") != hashlib.sha256(cycle_path.read_bytes()).hexdigest():
        raise ValueError("world-model full-system cycle SHA-256 mismatch")
    if binding.get("run_id") != cycle.get("run_id"):
        raise ValueError("world-model full-system cycle run_id mismatch")
    if world.get("summary", {}).get("downstream_hash_binding_ready") is not True:
        raise ValueError("world-model downstream hash binding is not ready")


__all__ = ["verify_world_model_cycle_binding"]
