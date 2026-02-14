"""
Error Fixer - Автоматичне виправлення поширених помилок в проекті
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import ast
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorFixer:
    """Автоматичний виправник помилок в коді"""
    
    def __init__(self):
        self.fixes_applied = []
        self.error_patterns = {
            'unicode_quotes': r'[\u201c\u201d]',  # Криві лапки
            'unicode_chars': r'[\u2013\u2014\u2026]',  # Unicode символи
            'f_string_issues': r'f"[^"]*\{[^}]*\}[^"]*"',  # Проблеми з f-string
            'indentation_issues': r'^\s*\t',  # Tab відступи
        }
        
    def fix_common_syntax_errors(self, file_path: str) -> Dict[str, Any]:
        """Виправляє поширені синтаксичні помилки у файлі"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes = []
            
            # Виправлення Unicode лапок
            content, quote_fixes = self._fix_unicode_quotes(content)
            fixes.extend(quote_fixes)
            
            # Виправлення Unicode символів
            content, unicode_fixes = self._fix_unicode_chars(content)
            fixes.extend(unicode_fixes)
            
            # Виправлення f-string проблем
            content, fstring_fixes = self._fix_f_string_issues(content)
            fixes.extend(fstring_fixes)
            
            # Виправлення відступів
            content, indent_fixes = self._fix_indentation(content)
            fixes.extend(indent_fixes)
            
            # Зберігаємо виправлений файл
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"[ErrorFixer] Fixed {len(fixes)} issues in {file_path}")
                self.fixes_applied.extend(fixes)
            
            return {
                'file': file_path,
                'fixes_applied': len(fixes),
                'fixes': fixes,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"[ErrorFixer] Error fixing {file_path}: {e}")
            return {
                'file': file_path,
                'fixes_applied': 0,
                'fixes': [],
                'success': False,
                'error': str(e)
            }
    
    def _fix_unicode_quotes(self, content: str) -> Tuple[str, List[str]]:
        """Виправляє Unicode лапки"""
        fixes = []
        
        # Заміна кривих лапок на нормальні
        content = re.sub(self.error_patterns['unicode_quotes'], '"', content)
        fixes.append("Fixed Unicode quotes")
        
        return content, fixes
    
    def _fix_unicode_chars(self, content: str) -> Tuple[str, List[str]]:
        """Виправляє Unicode символи"""
        fixes = []
        
        # Заміна тире на звичайне
        content = re.sub(self.error_patterns['unicode_chars'], '-', content)
        fixes.append("Fixed Unicode characters")
        
        return content, fixes
    
    def _fix_f_string_issues(self, content: str) -> Tuple[str, List[str]]:
        """Виправляє проблеми з f-string"""
        fixes = []
        
        # Виправлення екранованих лапок в f-string
        content = re.sub(r'f"[^"]*\\"[^"]*\\"[^"]*"', lambda m: m.group(0).replace('\\"', '"'), content)
        fixes.append("Fixed f-string quote escaping")
        
        return content, fixes
    
    def _fix_indentation(self, content: str) -> Tuple[str, List[str]]:
        """Виправляє відступи"""
        fixes = []
        
        # Заміна табів на пробіли
        content = re.sub(self.error_patterns['indentation_issues'], '    ', content)
        fixes.append("Fixed indentation (tabs to spaces)")
        
        return content, fixes
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Аналізує якість data і повертає рекомендації"""
        
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'issues': [],
            'recommendations': []
        }
        
        # Перевірка на пропущені значення
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            analysis['issues'].append(f"Missing values in {len(missing_cols)} columns")
            analysis['recommendations'].append(f"Consider imputation for: {missing_cols[:5]}")
        
        # Перевірка на дублікати
        if analysis['duplicate_rows'] > 0:
            analysis['issues'].append(f"Found {analysis['duplicate_rows']} duplicate rows")
            analysis['recommendations'].append("Remove duplicate rows")
        
        # Перевірка типів data
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            analysis['issues'].append(f"Found {len(object_cols)} object columns")
            analysis['recommendations'].append(f"Review object columns: {list(object_cols)[:3]}")
        
        return analysis
    
    def create_error_report(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Створює звіт про помилки"""
        
        report = {
            'summary': {
                'total_issues': sum(len(issues) for issues in scan_results.values() if isinstance(issues, list)),
                'categories_analyzed': len(scan_results),
                'most_critical': self._find_most_critical(scan_results),
            },
            'priority_fixes': self._get_priority_fixes(scan_results),
            'recommendations': self._get_recommendations(scan_results),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        return report
    
    def _find_most_critical(self, scan_results: Dict[str, Any]) -> List[str]:
        """Знаходить найкритичніші проблеми"""
        critical = []
        
        if 'syntax_errors' in scan_results and scan_results['syntax_errors']:
            critical.append('syntax_errors')
        
        if 'import_errors' in scan_results and scan_results['import_errors']:
            critical.append('import_errors')
        
        return critical
    
    def _get_priority_fixes(self, scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Отримує пріоритетні виправлення"""
        fixes = []
        
        # Синтаксичні помилки - найвищий пріоритет
        if 'syntax_errors' in scan_results:
            for error in scan_results['syntax_errors'][:5]:
                fixes.append({
                    'priority': 'HIGH',
                    'type': 'syntax',
                    'file': error.get('file', 'unknown'),
                    'description': f"Line {error.get('line', '?')}: {error.get('error', 'unknown')}"
                })
        
        # Помилки імпорту - високий пріоритет
        if 'import_errors' in scan_results:
            for error in scan_results['import_errors'][:5]:
                fixes.append({
                    'priority': 'HIGH',
                    'type': 'import',
                    'file': error.get('file', 'unknown'),
                    'description': f"Cannot import {error.get('import_name', 'unknown')}"
                })
        
        return fixes
    
    def _get_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Отримує рекомендації"""
        recommendations = []
        
        total_issues = sum(len(issues) for issues in scan_results.values() if isinstance(issues, list))
        
        if total_issues > 1000:
            recommendations.append("Consider refactoring large files with many issues")
        
        if scan_results.get('syntax_errors', []):
            recommendations.append("Fix syntax errors first - they prevent code from running")
        
        if scan_results.get('import_errors', []):
            recommendations.append("Check import paths and missing dependencies")
        
        recommendations.append("Set up automated testing to catch errors early")
        recommendations.append("Use linting tools to maintain code quality")
        
        return recommendations
