"""
Комплексний аудитор allх модулandв проекту
"""

import os
import ast
import importlib.util
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ComprehensiveAuditor:
    """Комплексний аудитор for аналandwithу allх модулandв"""
    
    def __init__(self):
        self.project_root = 'c:/trading_project'
        self.issues = {
            'syntax_errors': [],
            'import_errors': [],
            'undefined_variables': [],
            'performance_issues': [],
            'security_issues': [],
            'architecture_issues': [],
            'data_flow_issues': [],
            'code_quality_issues': [],
            'documentation_issues': [],
            'testing_issues': []
        }
        self.module_stats = {}
        self.recommendations = []
        
    def scan_all_modules(self) -> Dict[str, Any]:
        """Повний сканування allх модулandв"""
        
        logger.info("Starting comprehensive audit...")
        
        # 1. Сканування структури проекту
        project_structure = self._scan_project_structure()
        
        # 2. Аналandwith Python fileandв
        python_files = self._find_python_files()
        
        for file_path in python_files:
            self._analyze_python_file(file_path)
        
        # 3. Аналandwith конфandгурацandйних fileandв
        config_analysis = self._analyze_config_files()
        
        # 4. Аналandwith data
        data_analysis = self._analyze_data_files()
        
        # 5. Аналandwith forлежностей
        dependency_analysis = self._analyze_dependencies()
        
        # 6. Аналandwith архandтектури
        architecture_analysis = self._analyze_architecture()
        
        # 7. Геnotрацandя рекомендацandй
        self._generate_recommendations()
        
        return {
            'project_structure': project_structure,
            'module_stats': self.module_stats,
            'issues': self.issues,
            'config_analysis': config_analysis,
            'data_analysis': data_analysis,
            'dependency_analysis': dependency_analysis,
            'architecture_analysis': architecture_analysis,
            'recommendations': self.recommendations,
            'summary': self._generate_summary()
        }
    
    def _scan_project_structure(self) -> Dict[str, Any]:
        """Сканування структури проекту"""
        
        structure = {
            'directories': [],
            'python_files': [],
            'config_files': [],
            'data_files': [],
            'test_files': [],
            'documentation_files': []
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # Пропускаємо .venv and andншand системнand папки
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            rel_path = os.path.relpath(root, self.project_root)
            
            if rel_path != '.':
                structure['directories'].append(rel_path)
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, self.project_root)
                
                if file.endswith('.py'):
                    structure['python_files'].append(rel_file_path)
                elif file.endswith(('.json', '.yaml', '.yml', '.env', '.ini')):
                    structure['config_files'].append(rel_file_path)
                elif file.endswith(('.parquet', '.csv', '.pkl', '.joblib')):
                    structure['data_files'].append(rel_file_path)
                elif 'test' in file.lower() or file.startswith('test_'):
                    structure['test_files'].append(rel_file_path)
                elif file.endswith(('.md', '.txt', '.rst')):
                    structure['documentation_files'].append(rel_file_path)
        
        return structure
    
    def _find_python_files(self) -> List[str]:
        """Find all Python fileи"""
        
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_python_file(self, file_path: str):
        """Аналandwith Python fileу"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Баwithова сandтистика
            self.module_stats[file_path] = {
                'lines': len(content.splitlines()),
                'size': len(content),
                'imports': [],
                'functions': [],
                'classes': [],
                'issues': []
            }
            
            # Парсинг AST
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path)
            except SyntaxError as e:
                self.issues['syntax_errors'].append({
                    'file': file_path,
                    'line': e.lineno,
                    'error': str(e),
                    'severity': 'critical'
                })
            
            # Аналandwith andмпортandв
            self._analyze_imports(content, file_path)
            
            # Аналandwith якостand codeу
            self._analyze_code_quality(content, file_path)
            
            # Аналandwith беwithпеки
            self._analyze_security(content, file_path)
            
        except Exception as e:
            self.issues['import_errors'].append({
                'file': file_path,
                'error': f"Failed to read file: {str(e)}",
                'severity': 'high'
            })
    
    def _analyze_ast(self, tree: ast.AST, file_path: str):
        """Аналandwith AST for withнаходження функцandй and класandв"""
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.module_stats[file_path]['functions'].append(node.name)
                
                # Перевandрка на довгand функцandї
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    lines = node.end_lineno - node.lineno
                    if lines > 50:
                        self.issues['code_quality_issues'].append({
                            'file': file_path,
                            'line': node.lineno,
                            'issue': f'Long function {node.name} ({lines} lines)',
                            'severity': 'medium'
                        })
            
            elif isinstance(node, ast.ClassDef):
                self.module_stats[file_path]['classes'].append(node.name)
                
                # Перевandрка на великand класи
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    self.issues['code_quality_issues'].append({
                        'file': file_path,
                        'line': node.lineno,
                        'issue': f'Large class {node.name} ({len(methods)} methods)',
                        'severity': 'medium'
                    })
    
    def _analyze_imports(self, content: str, file_path: str):
        """Аналandwith andмпортandв"""
        
        lines = content.splitlines()
        imports = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
                
                # Перевandрка на потенцandйнand problemsи
                if 'import *' in line:
                    self.issues['code_quality_issues'].append({
                        'file': file_path,
                        'line': i,
                        'issue': 'Wildcard import detected',
                        'severity': 'medium'
                    })
                
                # Перевandрка на циклandчнand andмпорти (спрощена)
                if 'from .' in line and 'import' in line:
                    self.issues['import_errors'].append({
                        'file': file_path,
                        'line': i,
                        'issue': 'Potential circular import',
                        'severity': 'medium'
                    })
        
        self.module_stats[file_path]['imports'] = imports
    
    def _analyze_code_quality(self, content: str, file_path: str):
        """Аналandwith якостand codeу"""
        
        lines = content.splitlines()
        
        # Перевandрка на довгand рядки
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                self.issues['code_quality_issues'].append({
                    'file': file_path,
                    'line': i,
                    'issue': f'Long line ({len(line)} chars)',
                    'severity': 'low'
                })
        
        for i, line in enumerate(lines, 1):
                self.issues['code_quality_issues'].append({
                    'file': file_path,
                    'line': i,
                    'severity': 'low'
                })
        
        # Перевandрка на print statements
        for i, line in enumerate(lines, 1):
            if 'print(' in line and not line.strip().startswith('#'):
                self.issues['code_quality_issues'].append({
                    'file': file_path,
                    'line': i,
                    'issue': 'Print statement found (should use logging)',
                    'severity': 'low'
                })
        
        # Перевandрка на hardcoded values
        hardcoded_patterns = ['password', 'secret', 'key', 'token']
        for i, line in enumerate(lines, 1):
            for pattern in hardcoded_patterns:
                if pattern in line.lower() and '=' in line and not line.strip().startswith('#'):
                    self.issues['security_issues'].append({
                        'file': file_path,
                        'line': i,
                        'issue': f'Potential hardcoded {pattern}',
                        'severity': 'high'
                    })
    
    def _analyze_security(self, content: str, file_path: str):
        """Аналandwith беwithпеки"""
        
        lines = content.splitlines()
        
        # Перевandрка на notбеwithпечнand функцandї
        dangerous_functions = ['# eval(', '# exec(', 'os.system(', 'subprocess.call']
        
        for i, line in enumerate(lines, 1):
            for func in dangerous_functions:
                if func in line and not line.strip().startswith('#'):
                    severity = 'critical' if func in ['# eval(', '# exec('] else 'high'
                    self.issues['security_issues'].append({
                        'file': file_path,
                        'line': i,
                        'issue': f'Dangerous function: {func}',
                        'severity': severity
                    })
        
        # Перевandрка на SQL injection (спрощена)
        sql_patterns = ['execute(', 'cursor.execute']
        for i, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if pattern in line and '%' in line and not line.strip().startswith('#'):
                    self.issues['security_issues'].append({
                        'file': file_path,
                        'line': i,
                        'issue': 'Potential SQL injection',
                        'severity': 'high'
                    })
    
    def _analyze_config_files(self) -> Dict[str, Any]:
        """Аналandwith конфandгурацandйних fileandв"""
        
        config_analysis = {
            'found_configs': [],
            'issues': [],
            'recommendations': []
        }
        
        config_files = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith(('.json', '.yaml', '.yml', '.env', '.ini')):
                    config_files.append(os.path.join(root, file))
        
        for config_file in config_files:
            config_analysis['found_configs'].append(config_file)
            
            # Перевandрка на sensitive data
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if any(key in content.lower() for key in ['password', 'secret', 'key', 'token']):
                    config_analysis['issues'].append({
                        'file': config_file,
                        'issue': 'Sensitive data in config file',
                        'severity': 'high'
                    })
                
                # Перевandрка на вandдсутнandсть шифрування
                if config_file.endswith('.env') and 'SECRET' in content:
                    config_analysis['recommendations'].append({
                        'file': config_file,
                        'recommendation': 'Consider encrypting sensitive data'
                    })
            
            except Exception as e:
                config_analysis['issues'].append({
                    'file': config_file,
                    'issue': f'Failed to read config: {str(e)}',
                    'severity': 'medium'
                })
        
        return config_analysis
    
    def _analyze_data_files(self) -> Dict[str, Any]:
        """Аналandwith data"""
        
        data_analysis = {
            'found_data': [],
            'size_analysis': {},
            'issues': []
        }
        
        data_files = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith(('.parquet', '.csv', '.pkl', '.joblib')):
                    data_files.append(os.path.join(root, file))
        
        for data_file in data_files:
            try:
                size = os.path.getsize(data_file) / (1024 * 1024)  # MB
                data_analysis['found_data'].append(data_file)
                data_analysis['size_analysis'][data_file] = f"{size:.2f} MB"
                
                # Перевandрка на великand fileи
                if size > 1000:  # > 1GB
                    data_analysis['issues'].append({
                        'file': data_file,
                        'issue': f'Large data file ({size:.2f} MB)',
                        'severity': 'medium'
                    })
                
                # Спроба прочиandти for валandдацandї
                if data_file.endswith('.parquet'):
                    try:
                        df = pd.read_parquet(data_file)
                        if len(df) == 0:
                            data_analysis['issues'].append({
                                'file': data_file,
                                'issue': 'Empty dataset',
                                'severity': 'medium'
                            })
                    except Exception as e:
                        data_analysis['issues'].append({
                            'file': data_file,
                            'issue': f'Failed to read parquet: {str(e)}',
                            'severity': 'high'
                        })
            
            except Exception as e:
                data_analysis['issues'].append({
                    'file': data_file,
                    'issue': f'File access error: {str(e)}',
                    'severity': 'medium'
                })
        
        return data_analysis
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Аналandwith forлежностей"""
        
        dependency_analysis = {
            'requirements_files': [],
            'dependencies': {},
            'issues': []
        }
        
        # Пошук requirements.txt
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file == 'requirements.txt':
                    dependency_analysis['requirements_files'].append(os.path.join(root, file))
        
        # Аналandwith andмпортandв with Python fileandв
        all_imports = set()
        for file_path, stats in self.module_stats.items():
            for imp in stats['imports']:
                # Витягуємо наwithву модуля
                if 'import ' in imp:
                    parts = imp.split()
                    if len(parts) >= 2:
                        module = parts[1].split('.')[0]
                        all_imports.add(module)
        
        dependency_analysis['dependencies'] = list(all_imports)
        
        # Перевandрка на вandдсутнandсть requirements.txt
        if not dependency_analysis['requirements_files']:
            dependency_analysis['issues'].append({
                'issue': 'No requirements.txt found',
                'severity': 'medium',
                'recommendation': 'Create requirements.txt with all dependencies'
            })
        
        return dependency_analysis
    
    def _analyze_architecture(self) -> Dict[str, Any]:
        """Аналandwith архandтектури"""
        
        architecture_analysis = {
            'module_organization': {},
            'separation_of_concerns': {},
            'issues': [],
            'recommendations': []
        }
        
        # Аналandwith структури папок
        core_structure = {}
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'core')):
            if 'core' in root:
                rel_path = os.path.relpath(root, self.project_root)
                core_structure[rel_path] = {
                    'files': [f for f in files if f.endswith('.py')],
                    'subdirs': dirs
                }
        
        architecture_analysis['module_organization'] = core_structure
        
        # Перевandрка на роwithмandщення fileandв
        misplaced_files = []
        for file_path in self.module_stats.keys():
            if 'test' in file_path and 'tests' not in file_path:
                misplaced_files.append(file_path)
        
        if misplaced_files:
            architecture_analysis['issues'].append({
                'issue': f'Test files outside tests directory: {misplaced_files}',
                'severity': 'medium'
            })
        
        # Перевandрка на дублювання функцandональностand
        function_names = {}
        for file_path, stats in self.module_stats.items():
            for func in stats['functions']:
                if func not in function_names:
                    function_names[func] = []
                function_names[func].append(file_path)
        
        duplicates = {name: files for name, files in function_names.items() if len(files) > 1}
        if duplicates:
            architecture_analysis['issues'].append({
                'issue': f'Potential duplicate functions: {list(duplicates.keys())}',
                'severity': 'medium'
            })
        
        return architecture_analysis
    
    def _generate_recommendations(self):
        """Геnotрацandя рекомендацandй"""
        
        # Критичнand problemsи
        critical_issues = sum(len(issues) for issues in self.issues.values() 
                            if any(issue.get('severity') == 'critical' for issue in issues))
        
        if critical_issues > 0:
            self.recommendations.append({
                'priority': 'critical',
                'issue': f'{critical_issues} critical issues found',
                'action': 'Fix critical issues immediately',
                'impact': 'System stability'
            })
        
        # Проблеми беwithпеки
        if self.issues['security_issues']:
            self.recommendations.append({
                'priority': 'high',
                'issue': f'{len(self.issues["security_issues"])} security issues',
                'action': 'Review and fix security vulnerabilities',
                'impact': 'Data protection'
            })
        
        # Проблеми продуктивностand
        if self.issues['performance_issues']:
            self.recommendations.append({
                'priority': 'medium',
                'issue': f'{len(self.issues["performance_issues"])} performance issues',
                'action': 'Optimize performance bottlenecks',
                'impact': 'System speed'
            })
        
        # Якandсть codeу
        if self.issues['code_quality_issues']:
            self.recommendations.append({
                'priority': 'medium',
                'issue': f'{len(self.issues["code_quality_issues"])} code quality issues',
                'action': 'Improve code quality and maintainability',
                'impact': 'Long-term maintenance'
            })
        
        # Докуменandцandя
        total_files = len(self.module_stats)
        if total_files > 0:
            doc_files = 0
            for file_path, stats in self.module_stats.items():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:
                            doc_files += 1
                except:
                    pass
            
            doc_ratio = doc_files / total_files
            
            if doc_ratio < 0.5:
                self.recommendations.append({
                    'priority': 'low',
                    'issue': f'Low documentation coverage ({doc_ratio:.1%})',
                    'action': 'Add documentation to modules',
                    'impact': 'Code understanding'
                })
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Геnotрацandя пandдсумку"""
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for issues in self.issues.values():
            for issue in issues:
                severity = issue.get('severity', 'medium')
                severity_counts[severity] += 1
        
        return {
            'total_files_analyzed': len(self.module_stats),
            'total_issues': total_issues,
            'severity_breakdown': severity_counts,
            'health_score': max(0, 100 - (severity_counts['critical'] * 10 + 
                                        severity_counts['high'] * 5 + 
                                        severity_counts['medium'] * 2 + 
                                        severity_counts['low'] * 1)),
            'recommendations_count': len(self.recommendations)
        }


def run_comprehensive_audit():
    """Запуск комплексного аудиту"""
    
    auditor = ComprehensiveAuditor()
    results = auditor.scan_all_modules()
    
    return results


if __name__ == "__main__":
    logger.info("=== COMPREHENSIVE PROJECT AUDIT ===")
    
    results = run_comprehensive_audit()
    
    # Вивandд реwithульandтandв
    summary = results['summary']
    logger.info(f"\nFiles analyzed: {summary['total_files_analyzed']}")
    logger.info(f"Total issues: {summary['total_issues']}")
    logger.info(f"Health score: {summary['health_score']}/100")
    
    logger.info(f"\nSeverity breakdown:")
    for severity, count in summary['severity_breakdown'].items():
        if count > 0:
            logger.info(f"  {severity}: {count}")
    
    logger.info(f"\nTop recommendations:")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        logger.info(f"{i}. [{rec['priority'].upper()}] {rec['issue']}")
        logger.info(f"   Action: {rec['action']}")
    
    # Збереження реwithульandтandв
    output_path = 'c:/trading_project/audit_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nDetailed results saved to: {output_path}")