# src/models/model_selector/selector.py

from typing import Any, cast

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("FingerprintModelSelector")

class SmartModelSelector:  # Keep name for backward compatibility
    """
    Advanced model selector that uses Context Fingerprints to identify the best model
    for a given market state. Implements similarity-based search, consensus strategies,
    and a reward system for Actor-Critic improvement.

    Note: This is the fingerprint-based selector. For full context analysis,
    use SmartModelSelector from smart_selector.py
    """

    def __init__(self, fallback: str = "lightgbm"):
        self.fallback = fallback
        self.logger = logger
        self.reward_history: list[dict[str, Any]] = []

    def select_best_model(self, context_fingerprint: str, arena_leaderboard: dict[str, Any]) -> str:
        """
        Selects the best model based on 'points' or 'win_rate' for a specific fingerprint.
        If no exact match, performs a similarity check on the '1|0|-1' string bits.
        """
        try:
            # 1. Exact match check
            if context_fingerprint in arena_leaderboard:
                models_data = arena_leaderboard[context_fingerprint]
                # Select based on points, then win_rate
                best_model = max(models_data.keys(),
                                key=lambda m: (models_data[m].get('points', 0),
                                               models_data[m].get('win_rate', 0)))
                self.logger.info(f"Exact match for '{context_fingerprint}'. Selected: {best_model}")
                return cast(str, best_model)

            # 2. Similarity check (bit/state matching)
            target_bits = context_fingerprint.split('|')
            best_sim_fp = None
            max_matches = -1

            for fp in arena_leaderboard.keys():
                fp_bits = fp.split('|')
                if len(fp_bits) != len(target_bits):
                    continue

                matches = sum(1 for i in range(len(target_bits)) if target_bits[i] == fp_bits[i])
                if matches > max_matches:
                    max_matches = matches
                    best_sim_fp = fp

            # Require at least 50% match for fuzzy selection
            if best_sim_fp and max_matches >= (len(target_bits) / 2):
                models_data = arena_leaderboard[best_sim_fp]
                best_model = max(models_data.keys(),
                                key=lambda m: (models_data[m].get('points', 0),
                                               models_data[m].get('win_rate', 0)))
                self.logger.info(f"Fuzzy match '{best_sim_fp}' ({max_matches} bits). Selected: {best_model}")
                return cast(str, best_model)

            self.logger.warning(f"No reliable match for '{context_fingerprint}'. Fallback: {self.fallback}")
            return self.fallback

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Selection error: {e}", exc_info=True)
            return self.fallback

    def get_consensus_strategy(self, heavy_model: Any, light_model: Any, data: Any) -> dict[str, Any]:
        """
        Gets predictions from Heavy and Light models and validates direction.
        Returns CONFIRMED if they agree, or DIVERGENCE_WARNING if they clash.
        """
        try:
            h_pred = heavy_model.predict(data)
            l_pred = light_model.predict(data)

            # Simple direction check (sign)
            h_dir = self._get_direction(h_pred)
            l_dir = self._get_direction(l_pred)

            if h_dir == l_dir and h_dir != 0:
                confidence = self._calculate_consensus_confidence(h_pred, l_pred)
                action = self._get_action_from_direction(h_dir)
                self.logger.info(f"Consensus CONFIRMED: Heavy({h_pred:.4f}) and Light({l_pred:.4f}) agree.")
                return {
                    "status": "CONFIRMED",
                    "action": action,
                    "confidence": confidence,
                    "avg_pred": (h_pred + l_pred) / 2
                }

            self.logger.warning(f"DIVERGENCE_WARNING: Heavy({h_pred:.4f}) vs Light({l_pred:.4f}).")
            return {
                "status": "DIVERGENCE_WARNING",
                "action": "NEUTRAL",
                "confidence": 0.3,
                "suggestion": "Reduce position or skip trade"
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Consensus calculation failed: {e}")
            return {"status": "ERROR", "action": "SKIP"}

    def _get_direction(self, prediction: float) -> int:
        """Extract direction from prediction value"""
        if prediction > 0:
            return 1
        elif prediction < 0:
            return -1
        else:
            return 0

    def _calculate_consensus_confidence(self, h_pred: float, l_pred: float) -> float:
        """Calculate confidence based on prediction difference"""
        if abs(h_pred - l_pred) < 0.02:
            return 0.95
        else:
            return 0.85

    def _get_action_from_direction(self, direction: int) -> str:
        """Convert direction to action"""
        if direction > 0:
            return "BUY"
        else:
            return "SELL"

    def calculate_reward(self, predicted_direction: int, actual_direction: int, was_consensus: bool, critic_warned: bool) -> dict[str, float]:
        """
        Assigns rewards based on outcome and logic used.
        """
        reward = 0.0
        details = {}

        is_correct = (predicted_direction == actual_direction)

        if is_correct:
            reward += 1.0
            if was_consensus:
                reward += 0.5  # Premium Reward for confirmed consensus
                details["bonus"] = "Consensus Premium"
        else:
            reward -= 1.0
            # If Critic warned about divergence and it indeed resulted in a loss/flip
            if critic_warned:
                reward += 0.8 # Reward the Critic for correctly flagging risk
                details["critic_bonus"] = "Risk Mitigation Reward"

        result = {"total_reward": round(reward, 2), "details": details}
        self.reward_history.append(result)
        self.logger.info(f"Decision Audit - Reward: {reward} | Consensus: {was_consensus} | Correct: {is_correct}")

        return cast(dict[str, float], result)
