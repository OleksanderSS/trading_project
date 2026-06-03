"""Exposure calculation helper used by RiskManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.risk.risk_manager import Position


class ExposureCalculator:
    def __init__(
        self,
        portfolio_value: float,
        max_position_exposure: float = 0.25,
        max_sector_exposure: float = 0.40,
        max_total_exposure: float = 1.0,
    ):
        self.portfolio_value = portfolio_value
        self.max_position_exposure = max_position_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_total_exposure = max_total_exposure

    def calculate_position_exposure(self, position: Position) -> float:
        return abs(position.value) / self.portfolio_value if self.portfolio_value > 0 else 1.0

    def check_position_exposure(self, position: Position) -> tuple[bool, str]:
        exposure = self.calculate_position_exposure(position)
        if exposure > self.max_position_exposure:
            message = (
                f"Position exposure {exposure:.1%} exceeds limit {self.max_position_exposure:.1%} for {position.ticker}"
            )
            return False, message
        return True, "OK"

    def check_sector_exposure(self, positions: list[Position]) -> tuple[bool, dict[str, float]]:
        sector_exposure: dict[str, float] = {}
        for pos in positions:
            sector = pos.sector
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(pos.value)

        violations = []
        for sector, exposure_value in sector_exposure.items():
            exposure_pct = exposure_value / self.portfolio_value if self.portfolio_value > 0 else 1.0
            if exposure_pct > self.max_sector_exposure:
                violations.append(sector)

        return len(violations) == 0, sector_exposure

    def check_total_exposure(self, positions: list[Position]) -> tuple[bool, float]:
        total_exposure_value = sum(abs(pos.value) for pos in positions)
        total_exposure_pct = total_exposure_value / self.portfolio_value if self.portfolio_value > 0 else 1.0
        return total_exposure_pct <= self.max_total_exposure, total_exposure_pct
