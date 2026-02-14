"""
Комплексний скаnotр errors and слабких мandсць проекту
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ProjectErrorScanner:
    """Скаnotр errors and слабких мandсць у всьому проектand"""
    
    def __init__(self, project_root: str = "c:/trading_project"):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.suggestions = []
        
    def scan_entire_project(self) -> Dict[str, Any]:
        """Повний сканування проекту"""
        
        results = {
            'syntax_errors': self._scan_syntax_errors(),
            'import_errors': self._scan_import_errors(),
            'undefined_variables': self._scan_undefined_variables(),
            'performance_issues': self._scan_performance_issues(),
            'code_quality': self._scan_code_quality(),
            'security_issues': self._scan_security_issues(),
            'architecture_issues': self._scan_architecture_issues(),
            'data_flow_issues': self._scan_data_flow_issues(),
            'summary': {}
        }
        
        # Пandдрахунок forгальної сandтистики
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _scan_syntax_errors(self) -> List[Dict[str, Any]]:
        """Сканування синandксичних errors"""
        
        syntax_errors = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Перевandрка синandксису
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_errors.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': e.lineno,
                        'error': str(e),
                        'severity': 'critical'
                    })
                    
            except Exception as e:
                syntax_errors.append({
                    'file': str(file_path.relative_to(self.project_root)),
                    'error': f"File read error: {e}",
                    'severity': 'high'
                })
        
        return syntax_errors
    
    def _scan_import_errors(self) -> List[Dict[str, Any]]:
        """Сканування errors andмпортandв"""
        
        import_errors = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Аналandwith andмпортandв
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('.'):
                                # Перевandрка вandдносних andмпортandв
                                import_errors.append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'import': alias.name,
                                    'issue': 'Relative import',
                                    'severity': 'medium'
                                })
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and not node.module.startswith('.'):
                            # Перевandрка абсолютних andмпортandв
                            import_errors.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'line': node.lineno,
                                'import': node.module,
                                'issue': 'Absolute import check needed',
                                'severity': 'low'
                            })
                            
            except Exception as e:
                continue
        
        return import_errors
    
    def _scan_undefined_variables(self) -> List[Dict[str, Any]]:
        """Сканування notвиwithначених withмandнних"""
        
        undefined_vars = []
        python_files = list(self.project_root.rglob("*.py"))
        
        # Список потенцandйних problemsних withмandнних
        problematic_vars = ['X_test', 'y_test', 'target_col', 'task_type']
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    # Перевandрка викорисandння X_test/y_test
                    if 'X_test' in line or 'y_test' in line:
                        undefined_vars.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'variable': 'X_test/y_test',
                            'code': line.strip(),
                            'issue': 'Potential undefined variable',
                            'severity': 'high'
                        })
                    
                    # Перевandрка викорисandння target_col беwith виvalues
                    if 'target_col' in line and 'target_col =' not in line:
                        undefined_vars.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'variable': 'target_col',
                            'code': line.strip(),
                            'issue': 'target_col used without definition',
                            'severity': 'high'
                        })
                        
            except Exception:
                continue
        
        return undefined_vars
    
    def _scan_performance_issues(self) -> List[Dict[str, Any]]:
        """Сканування problems продуктивностand"""
        
        performance_issues = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # Перевandрка на notефективнand операцandї
                    if 'pd.read_csv' in line and 'chunksize' not in line:
                        performance_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'issue': 'Large file read without chunking',
                            'code': line.strip(),
                            'severity': 'medium'
                        })
                    
                    # Перевandрка на цикли в Pandas
                    if 'for' in line and 'iterrows()' in line:
                        performance_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'issue': 'iterrows() usage (slow)',
                            'code': line.strip(),
                            'severity': 'medium'
                        })
                    
                    # Перевandрка на глобальнand withмandннand
                    if 'global ' in line:
                        performance_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'issue': 'Global variable usage',
                            'code': line.strip(),
                            'severity': 'low'
                        })
                        
            except Exception:
                continue
        
        return performance_issues
    
    def _scan_code_quality(self) -> List[Dict[str, Any]]:
        """Сканування якостand codeу"""
        
        quality_issues = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Довгand рядки
                    if len(line) > 120:
                        quality_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'issue': 'Long line (>120 chars)',
                            'length': len(line),
                            'severity': 'low'
                        })
                    
                        quality_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'code': line,
                            'severity': 'medium'
                        })
                    
                    # Hardcoded values
                    if re.search(r'\b\d{2,}\b', line) and '=' in line:
                        quality_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'issue': 'Hardcoded number',
                            'code': line,
                            'severity': 'low'
                        })
                        
            except Exception:
                continue
        
        return quality_issues
    
    def _scan_security_issues(self) -> List[Dict[str, Any]]:
        """Сканування problems беwithпеки"""
        
        security_issues = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Перевandрка на notбеwithпечнand функцandї
                dangerous_patterns = [
                    ('# eval(', 'Use of # eval()'),
                    ('# exec(', 'Use of # exec()'),
                    ('os.system', 'Use of os.system'),
                    ('subprocess.call', 'Use of subprocess.call'),
                    ('input(', 'Use of input()'),
                    ('password', 'Password in code'),
                    ('api_key', 'API key in code')
                ]
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    for pattern, issue in dangerous_patterns:
                        if pattern in line:
                            security_issues.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'line': i,
                                'issue': issue,
                                'code': line.strip(),
                                'severity': 'high' if 'eval' in pattern or 'exec' in pattern else 'medium'
                            })
                            
            except Exception:
                continue
        
        return security_issues
    
    def _scan_architecture_issues(self) -> List[Dict[str, Any]]:
        """Сканування архandтектурних problems"""
        
        arch_issues = []
        
        # Перевandрка структури папок
        core_path = self.project_root / 'core'
        if not core_path.exists():
            arch_issues.append({
                'issue': 'Missing core directory',
                'severity': 'critical'
            })
        
        # Перевandрка наявностand ключових модулandв
        key_modules = [
            'core/stages/stage_1_collectors_layer.py',
            'core/stages/stage_2_enrichment.py',
            'core/stages/stage_3_utils.py',
            'core/stages/stage_4_modeling.py',
            'config/feature_config.py'
        ]
        
        for module in key_modules:
            module_path = self.project_root / module
            if not module_path.exists():
                arch_issues.append({
                    'issue': f'Missing key module: {module}',
                    'severity': 'high'
                })
        
        # Перевandрка дублandкатandв fileandв
        file_counts = {}
        for file_path in self.project_root.rglob("*.py"):
            name = file_path.name
            if name not in file_counts:
                file_counts[name] = []
            file_counts[name].append(file_path)
        
        for name, paths in file_counts.items():
            if len(paths) > 1:
                arch_issues.append({
                    'issue': f'Duplicate file: {name}',
                    'files': [str(p.relative_to(self.project_root)) for p in paths],
                    'severity': 'medium'
                })
        
        return arch_issues
    
    def _scan_data_flow_issues(self) -> List[Dict[str, Any]]:
        """Сканування problems потоку data"""
        
        data_flow_issues = []
        
        # Перевandрка наявностand data
        data_files = [
            'data/stages/merged_full.parquet',
            'data/stages/merged_full_clean.parquet'
        ]
        
        for data_file in data_files:
            file_path = self.project_root / data_file
            if not file_path.exists():
                data_flow_issues.append({
                    'issue': f'Missing data file: {data_file}',
                    'severity': 'high'
                })
        
        # Перевandрка конфandгурацandї
        config_files = [
            'config/config.py',
            'config/feature_config.py'
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Перевandрка на порожнand конфandгурацandї
                    if 'TICKERS' in content and '{}' in content:
                        data_flow_issues.append({
                            'issue': f'Empty configuration in {config_file}',
                            'severity': 'medium'
                        })
                        
                except Exception:
                    continue
        
        return data_flow_issues
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрацandя withвandту"""
        
        summary = {
            'total_issues': 0,
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0,
            'categories': {}
        }
        
        for category, issues in results.items():
            if category == 'summary':
                continue
                
            if isinstance(issues, list):
                summary['categories'][category] = len(issues)
                summary['total_issues'] += len(issues)
                
                for issue in issues:
                    severity = issue.get('severity', 'low')
                    if severity == 'critical':
                        summary['critical_issues'] += 1
                    elif severity == 'high':
                        summary['high_issues'] += 1
                    elif severity == 'medium':
                        summary['medium_issues'] += 1
                    else:
                        summary['low_issues'] += 1
        
        return summary


def run_project_scan() -> Dict[str, Any]:
    """Запуск сканування проекту"""
    
    scanner = ProjectErrorScanner()
    results = scanner.scan_entire_project()
    
    return results


def print_scan_results(results: Dict[str, Any]):
    """Друк реwithульandтandв сканування"""
    
    summary = results['summary']
    
    logger.info("=== PROJECT SCAN RESULTS ===")
    logger.info(f"Total issues: {summary['total_issues']}")
    logger.info(f"Critical: {summary['critical_issues']}")
    logger.info(f"High: {summary['high_issues']}")
    logger.info(f"Medium: {summary['medium_issues']}")
    logger.info(f"Low: {summary['low_issues']}")
    
    logger.info("\n=== ISSUES BY CATEGORY ===")
    for category, count in summary['categories'].items():
        if count > 0:
            logger.info(f"{category}: {count}")
    
    # Критичнand problemsи
    logger.info("\n=== CRITICAL ISSUES ===")
    for category, issues in results.items():
        if isinstance(issues, list):
            critical = [i for i in issues if i.get('severity') == 'critical']
            if critical:
                logger.info(f"\n{category.upper()}:")
                for issue in critical[:5]:  # Покаwithуємо першand 5
                    logger.info(f"  - {issue.get('issue', 'Unknown')}")
                    if 'file' in issue:
                        logger.info(f"    File: {issue['file']}")
    
    # Високand problemsи
    logger.info("\n=== HIGH PRIORITY ISSUES ===")
    for category, issues in results.items():
        if isinstance(issues, list):
            high = [i for i in issues if i.get('severity') == 'high']
            if high:
                logger.info(f"\n{category.upper()}:")
                for issue in high[:3]:  # Покаwithуємо першand 3
                    logger.info(f"  - {issue.get('issue', 'Unknown')}")
                    if 'file' in issue:
                        logger.info(f"    File: {issue['file']}")


if __name__ == "__main__":
    results = run_project_scan()
    print_scan_results(results)