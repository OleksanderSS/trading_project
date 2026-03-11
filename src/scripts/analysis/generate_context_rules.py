#!/usr/bin/env python3
"""
Runner script for generating context rules from historical data analysis.
This script initializes the necessary components and executes the rule generation
workflow defined in the core analysis module.
"""

import logging
import os
import sys
import argparse

# Add project root to the Python path
# This allows for direct execution of the script, making imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.core.analysis.rule_generator import ContextRuleGenerator
from src.utils.config_manager import UnifiedConfigManager
# Corrected import path
from src.core.data.data_manager import DataManager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config_path: str):
    """
    Main function to initialize components and run the rule generation process.
    
    Args:
        config_path (str): Path to the main configuration file.
    """
    logger.info("Initializing context rule generation process.")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at '{config_path}'. Aborting.")
        return

    try:
        # --- Initialization ---
        config_manager = UnifiedConfigManager(config_path=config_path)
        data_manager = DataManager(config_manager) 
        
        rule_generator = ContextRuleGenerator(config_manager, data_manager)
        
        # --- Path Resolution ---
        # The core component defines a relative path; the runner makes it absolute.
        output_path_relative = rule_generator.analysis_config.get('output_path')
        if not output_path_relative:
            logger.error("`output_path` not found in the 'context_rule_generation' config. Aborting.")
            return

        # Make the path absolute relative to the project root
        output_path_absolute = os.path.join(project_root, output_path_relative)
        
        # Override the path in the generator's configuration dictionary
        rule_generator.analysis_config['output_path'] = output_path_absolute
        
        logger.info(f"Output rules will be saved to: {output_path_absolute}")

        # --- Execution ---
        rule_generator.run_analysis()
        
        logger.info("Context rule generation process completed successfully.")

    except ValueError as e:
        logger.error(f"Configuration or value error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Context Rules for Market Analysis")
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to the configuration file (relative to project root).'
    )
    args = parser.parse_args()
    
    # Construct the full path to the config file
    full_config_path = os.path.join(project_root, args.config)
    
    main(config_path=full_config_path)
