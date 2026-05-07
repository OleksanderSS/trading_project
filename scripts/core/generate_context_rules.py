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

from src.devtools.rule_generator import ContextRuleGenerator
from src.config.unified_config_manager import get_current_config
from src.data.management.data_manager import DataManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ContextRuleGenerator")

def main():
    """
    Main function to initialize components and run the rule generation process.
    """
    logger.info("Initializing context rule generation process.")
    
    try:
        # --- Initialization ---
        config_manager = get_current_config()
        data_manager = DataManager(config_manager) 
        
        rule_generator = ContextRuleGenerator(config_manager, data_manager)
        
        # --- Path Resolution ---
        output_path_relative = rule_generator.analysis_config.get('output_path')
        if not output_path_relative:
            logger.error("`output_path` not found in the 'context_rule_generation' config. Aborting.")
            return

        # Use project root from DataManager
        output_path_absolute = Path(project_root) / output_path_relative
        
        # Override the path in the generator's configuration dictionary
        rule_generator.analysis_config['output_path'] = str(output_path_absolute)
        
        logger.info(f"Output rules will be saved to: {output_path_absolute}")

        # --- Execution ---
        rule_generator.run_analysis()
        
        logger.info("Context rule generation process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
