import json
import logging
from pathlib import Path

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class DynamicRouter:
    """
    Adjusts the weights of models in the Arena dynamically based on the current market context
    (e.g., sharp drops, high volatility) using pre-computed routing rules from pattern mining.
    """
    def __init__(self):
        self.config = UnifiedConfigManager()
        self.rules_path = Path(self.config.get('paths.processed_data', 'data/processed')) / 'routing_rules.json'
        self.rules = {}
        self.load_rules()
        self.audit_log_path = Path(self.config.get('paths.logs', 'logs')) / 'router_audit.jsonl'
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_rules(self):
        if self.rules_path.exists():
            try:
                with open(self.rules_path, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
                logger.info("Dynamic routing rules loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load routing rules: {e}")
        else:
            logger.warning(f"No routing rules found at {self.rules_path}. Operating in default mode.")

    def adjust_weights(self, model_predictions: dict, context_features: dict) -> dict:
        """
        Adjusts the implicit or explicit weights of the models based on the context.
        Returns a dictionary mapping model names to their weight multiplier.
        """
        weight_multipliers = {m: 1.0 for m in model_predictions.keys()}
        
        if not self.rules:
            return weight_multipliers
            
        is_sharp_drop = context_features.get('is_sharp_drop', False)
        # We also treat the 'regime' from ensemble as a sharp drop if it's 'sharp_drop'
        if context_features.get('regime') == 'sharp_drop':
            is_sharp_drop = True
            
        if is_sharp_drop and 'sharp_drop' in self.rules:
            drop_rules = self.rules['sharp_drop']
            target_mods = drop_rules.get('target_modifiers', {})
            hour_mods = drop_rules.get('hour_modifiers', {})
            day_mods = drop_rules.get('day_modifiers', {})
            trend_mods = drop_rules.get('trend_modifiers', {})
            prob_calibration = drop_rules.get('probability_calibration', {})
            
            # Extract basic context
            hour_str = str(context_features.get('hour_of_day', ''))
            day_str = str(context_features.get('day_of_week', ''))
            trend_str = str(context_features.get('trend_state', ''))
            
            # Fetch global context multipliers
            hour_mult = hour_mods.get(hour_str, 1.0)
            day_mult = day_mods.get(day_str, 1.0)
            trend_mult = trend_mods.get(trend_str, 1.0)
            
            for mname in weight_multipliers.keys():
                # Extract target from model name
                target_type = 'unknown'
                if '_target_' in mname:
                    parts = mname.split('_target_')[1].split('_')
                    target_type = '_'.join(parts[:-1])
                
                # Apply model-specific target modifier
                target_mult = target_mods.get(target_type, 1.0)
                
                # We could also use probabilities if available in context_features
                # For now, we apply global time/trend and specific target modifiers
                final_mult = target_mult * hour_mult * day_mult * trend_mult
                
                weight_multipliers[mname] *= final_mult
                    
            logger.info(f"Applied 'sharp_drop' routing rules. Base Mults: Hour={hour_mult}, Day={day_mult}")
            
            # Audit log for live verification
            try:
                import datetime
                audit_entry = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "context_features": context_features,
                    "multipliers": weight_multipliers
                }
                with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(audit_entry) + '\n')
            except Exception as e:
                logger.error(f"Failed to write router audit log: {e}")
            
        return weight_multipliers
