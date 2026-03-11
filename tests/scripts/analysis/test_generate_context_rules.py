#!/usr/bin/env python3
"""
Test suite for the context rule generation script.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Import the main function from the script
from src.scripts.analysis.generate_context_rules import main as generate_rules_main

class TestGenerateContextRules(unittest.TestCase):
    """Test cases for the rule generation script."""

    @patch('src.scripts.analysis.generate_context_rules.UnifiedConfigManager')
    @patch('src.scripts.analysis.generate_context_rules.DataManager')
    @patch('src.scripts.analysis.generate_context_rules.ContextRuleGenerator')
    @patch('os.path.exists')
    def test_main_success_flow(self, mock_path_exists, mock_RuleGenerator, mock_DataManager, mock_ConfigManager):
        """Test the successful execution of the main function."""
        # --- Setup Mocks ---
        mock_path_exists.return_value = True
        
        # Mock config manager
        mock_config_instance = MagicMock()
        mock_ConfigManager.return_value = mock_config_instance
        
        # Mock data manager
        mock_data_instance = MagicMock()
        mock_DataManager.return_value = mock_data_instance
        
        # Mock rule generator
        mock_generator_instance = MagicMock()
        mock_generator_instance.analysis_config = {'output_path': 'src/config/test_rules.yaml'}
        mock_RuleGenerator.return_value = mock_generator_instance

        # --- Execute ---
        config_path = 'fake_config.json'
        generate_rules_main(config_path)

        # --- Assertions ---
        # Verify that all components were initialized as expected
        mock_ConfigManager.assert_called_once_with(config_path=config_path)
        mock_DataManager.assert_called_once_with(mock_config_instance)
        mock_RuleGenerator.assert_called_once_with(mock_config_instance, mock_data_instance)
        
        # Verify that the analysis was run
        mock_generator_instance.run_analysis.assert_called_once()

        # Check if the output path was correctly resolved and set
        expected_abs_path = os.path.join(project_root, 'src/config/test_rules.yaml')
        self.assertEqual(mock_generator_instance.analysis_config['output_path'], expected_abs_path)

    @patch('src.scripts.analysis.generate_context_rules.logger')
    @patch('os.path.exists')
    def test_main_config_not_found(self, mock_path_exists, mock_logger):
        """Test the script's behavior when the config file is not found."""
        mock_path_exists.return_value = False
        
        config_path = 'non_existent_config.json'
        generate_rules_main(config_path)
        
        # Verify that an error was logged and the process was aborted
        mock_logger.error.assert_called_with(f"Configuration file not found at '{config_path}'. Aborting.")

if __name__ == "__main__":
    unittest.main()
