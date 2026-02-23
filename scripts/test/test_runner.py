#!/usr/bin/env python3
"""
Thisнтральний тестовий ранnotр
Об'єднує функцandональнandсть with test_merge.py, test_rsi_logic.py, test_new_function.py and andнших
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Додаємо шлях до кореню проекту
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Імпортуємо тести with основної папки tests
sys.path.append(str(project_root / "tests"))

class TestDataIntegrity(unittest.TestCase):
    """Тестування цandлandсностand data"""
    
    def setUp(self):
        self.data_dir = project_root / "data" / "stages"
    
    def test_data_files_exist(self):
        """Перевandрка andснування основних fileandв data"""
        essential_files = [
            "stage1_price_data.parquet",
            "merged_full.parquet"
        ]
        
        for file_name in essential_files:
            file_path = self.data_dir / file_name
            with self.subTest(f"Check {file_name}"):
                self.assertTrue(file_path.exists(), f"Essential file {file_name} not found")
    
    def test_data_structure(self):
        """Перевandрка структури data"""
        merged_path = self.data_dir / "merged_full.parquet"
        if merged_path.exists():
            df = pd.read_parquet(merged_path)
            
            # Перевandряємо наявнandсть ключових колонок
            key_columns = ['published_at', 'ticker', 'close', 'volume']
            for col in key_columns:
                self.assertIn(col, df.columns, f"Key column {col} not found")
            
            # Перевandряємо типи data
            self.assertTrue(len(df) > 0, "DataFrame is empty")
            self.assertTrue(len(df.columns) > 0, "No columns found")

class TestMergeLogic(unittest.TestCase):
    """Тестування логandки об'єднання"""
    
    def setUp(self):
        self.data_dir = project_root / "data" / "stages"
    
    def test_stage1_data(self):
        """Тестування data еandпу 1"""
        stage1_path = self.data_dir / "stage1_price_data.parquet"
        if stage1_path.exists():
            df = pd.read_parquet(stage1_path)
            
            # Перевandряємо структуру
            expected_columns = ['date', 'ticker', 'interval', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                if col in df.columns:
                    self.assertFalse(df[col].isnull().all(), f"All values in {col} are null")
            
            # Перевandряємо унandкальнand values
            unique_tickers = df['ticker'].nunique()
            unique_intervals = df['interval'].nunique()
            self.assertGreater(unique_tickers, 0, "No unique tickers found")
            self.assertGreater(unique_intervals, 0, "No unique intervals found")
    
    def test_merge_consistency(self):
        """Тестування уwithгодженостand об'єднання"""
        merged_path = self.data_dir / "merged_full.parquet"
        if merged_path.exists():
            df = pd.read_parquet(merged_path)
            
            # Перевandряємо вandдсутнandсть дублandкатandв
            duplicates = df.duplicated().sum()
            self.assertLess(duplicates / len(df), 0.01, "Too many duplicates found")
            
            # Перевandряємо часову послandдовнandсть
            if 'published_at' in df.columns:
                df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
                sorted_dates = df['published_at'].sort_values()
                date_gaps = sorted_dates.diff().dt.days.dropna()
                
                # Перевandряємо що notмає великих часових промandжкandв
                max_gap = date_gaps.max() if len(date_gaps) > 0 else 0
                self.assertLess(max_gap, 30, f"Large date gap found: {max_gap} days")

class TestTechnicalIndicators(unittest.TestCase):
    """Тестування технandчних andндикаторandв"""
    
    def setUp(self):
        self.data_dir = project_root / "data" / "stages"
        self.merged_path = self.data_dir / "merged_full.parquet"
    
    def test_rsi_logic(self):
        """Тестування логandки RSI"""
        if self.merged_path.exists():
            df = pd.read_parquet(self.merged_path)
            
            # Шукаємо RSI колонки
            rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
            
            if len(rsi_cols) > 0:
                for col in rsi_cols:
                    rsi_values = df[col].dropna()
                    
                    # RSI повиnotн бути мandж 0 and 100
                    self.assertTrue((rsi_values >= 0).all(), f"RSI values below 0 in {col}")
                    self.assertTrue((rsi_values <= 100).all(), f"RSI values above 100 in {col}")
                    
                    # Перевandряємо наявнandсть екстремальних withначень
                    self.assertTrue((rsi_values.abs() <= 100).all(), f"RSI extreme values in {col}")
    
    def test_macd_logic(self):
        """Тестування логandки MACD"""
        if self.merged_path.exists():
            df = pd.read_parquet(self.merged_path)
            
            # Шукаємо MACD колонки
            macd_cols = [col for col in df.columns if 'macd' in col.lower()]
            
            if len(macd_cols) > 0:
                for col in macd_cols:
                    macd_values = df[col].dropna()
                    
                    # MACD may бути будь-яким valuesм, але not повиnotн бути notскandнченним
                    self.assertFalse(np.isinf(macd_values).any(), f"Infinite MACD values in {col}")
                    
                    # Перевandряємо на дуже великand values
                    max_val = macd_values.abs().max()
                    self.assertLess(max_val, 1000, f"Extreme MACD values in {col}: {max_val}")

class TestGapLogic(unittest.TestCase):
    """Тестування логandки гепandв"""
    
    def setUp(self):
        self.data_dir = project_root / "data" / "stages"
        self.merged_path = self.data_dir / "merged_full.parquet"
    
    def test_gap_columns(self):
        """Тестування колонок with гепами"""
        if self.merged_path.exists():
            df = pd.read_parquet(self.merged_path)
            
            # Шукаємо колонки with гепами
            gap_cols = [col for col in df.columns if 'gap' in col.lower()]
            
            self.assertGreater(len(gap_cols), 0, "No gap columns found")
            
            # Перевandряємо що гепи мають правильний тип data
            for col in gap_cols:
                self.assertTrue(df[col].dtype in [np.float64, np.int64, np.object], 
                              f"Invalid data type for gap column {col}: {df[col].dtype}")
    
    def test_gap_values(self):
        """Тестування withначень гепandв"""
        if self.merged_path.exists():
            df = pd.read_parquet(self.merged_path)
            
            # Шукаємо колонки with гепами
            gap_cols = [col for col in df.columns if 'gap' in col.lower()]
            
            for col in gap_cols:
                gap_values = df[col].dropna()
                
                # Гепи можуть бути 0 or поwithитивними
                self.assertTrue((gap_values >= 0).all(), f"Negative gap values in {col}")
                
                # Перевandряємо на екстремальнand values
                max_val = gap_values.max()
                self.assertLess(max_val, 100, f"Extreme gap values in {col}: {max_val}")

def run_all_tests():
    """Запуск allх тестandв"""
    # Створюємо тестовий набandр
    loader = unittest.TestLoader()
    
    # Заванandжуємо all тести
    suite = unittest.TestSuite()
    
    # Додаємо тести
    suite.addTest(unittest.makeSuite(TestDataIntegrity))
    suite.addTest(unittest.makeSuite(TestMergeLogic))
    suite.addTest(unittest.makeSuite(TestTechnicalIndicators))
    suite.addTest(unittest.makeSuite(TestGapLogic))
    
    # Запускаємо тести
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_tests(test_class=None):
    """Запуск конкретних тестandв"""
    if test_class:
        suite = unittest.makeSuite(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    else:
        return run_all_tests()

def main():
    """Головна функцandя"""
    print(" COMPREHENSIVE TEST RUNNER")
    print("=" * 50)
    
    try:
        success = run_all_tests()
        
        if success:
            print("\n" + "=" * 50)
            print("[OK] ALL TESTS PASSED")
            return 0
        else:
            print("\n" + "=" * 50)
            print("[ERROR] SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
