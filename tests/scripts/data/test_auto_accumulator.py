#!/usr/bin/env python3
"""
Test suite for the AutoAccumulator script.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scripts.data.auto_accumulator import main as auto_accumulator_main

class TestAutoAccumulatorScript(unittest.TestCase):
    """Test cases for the auto_accumulator script."""

    @patch('src.scripts.data.auto_accumulator.AutoAccumulator')
    @patch('sys.argv', ['auto_accumulator.py', '--mode', 'cycle', '--group', 'tech'])
    def test_cycle_mode(self, mock_AutoAccumulator):
        """Test if the 'cycle' mode calls the correct method."""
        mock_instance = mock_AutoAccumulator.return_value
        auto_accumulator_main()
        mock_instance.run_accumulation_cycle.assert_called_once_with('tech')

    @patch('src.scripts.data.auto_accumulator.AutoAccumulator')
    @patch('sys.argv', ['auto_accumulator.py', '--mode', 'scheduled', '--hours', '12'])
    def test_scheduled_mode(self, mock_AutoAccumulator):
        """Test if the 'scheduled' mode calls the correct method."""
        mock_instance = mock_AutoAccumulator.return_value
        auto_accumulator_main()
        mock_instance.run_scheduled_accumulation.assert_called_once_with(12)

    @patch('src.scripts.data.auto_accumulator.AutoAccumulator')
    @patch('sys.argv', ['auto_accumulator.py', '--mode', 'continuous'])
    def test_continuous_mode(self, mock_AutoAccumulator):
        """Test if the 'continuous' mode calls the correct method."""
        mock_instance = mock_AutoAccumulator.return_value
        auto_accumulator_main()
        mock_instance.run_continuous_accumulation.assert_called_once()

    @patch('src.scripts.data.auto_accumulator.AutoAccumulator')
    @patch('sys.argv', ['auto_accumulator.py', '--report'])
    def test_report_mode(self, mock_AutoAccumulator):
        """Test if the '--report' flag calls the report generation method."""
        mock_instance = mock_AutoAccumulator.return_value
        mock_instance.get_accumulation_report.return_value = {
            'timestamp': '2023-10-27',
            'database_status': {'total_records': 1000},
            'configuration': {'sample_key': 'sample_value'},
            'recommendations': ['Run data cleaning']
        }
        auto_accumulator_main()
        mock_instance.get_accumulation_report.assert_called_once()

if __name__ == "__main__":
    unittest.main()
