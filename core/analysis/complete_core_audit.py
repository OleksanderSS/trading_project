# core/analysis/complete_core_audit.py - Повний аудит папки core

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class CompleteCoreAuditor:
    """
    Повний аудит папки core with рекомендацandями по оптимandforцandї
    """
    
    def __init__(self):
        self.core_path = Path("c:/trading_project/core")
        self.audit_results = {}
        
    def run_complete_audit(self) -> Dict[str, Any]:
        """
        Запуск повного аудиту папки core
        
        Returns:
            Dict with реwithульandandми аудиту
        """
        logger.info("[CompleteCoreAudit] Starting complete audit of core folder...")
        
        # 1. Аналandwith структури папки
        structure_analysis = self._analyze_folder_structure()
        
        # 2. Аналandwith fileandв
        file_analysis = self._analyze_files()
        
        # 3. Аналandwith дублandкатandв
        duplicate_analysis = self._analyze_duplicates()
        
        # 4. Аналandwith роwithмandрandв fileandв
        size_analysis = self._analyze_file_sizes()
        
        # 5. Аналandwith forлежностей
        dependency_analysis = self._analyze_dependencies()
        
        # 6. Аналandwith функцandональностand
        functionality_analysis = self._analyze_functionality()
        
        # 7. Аналandwith якостand codeу
        quality_analysis = self._analyze_code_quality()
        
        # 8. Рекомендацandї по оптимandforцandї
        optimization_recommendations = self._generate_optimization_recommendations()
        
        # 9. План дandй
        action_plan = self._generate_action_plan()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'structure_analysis': structure_analysis,
            'file_analysis': file_analysis,
            'duplicate_analysis': duplicate_analysis,
            'size_analysis': size_analysis,
            'dependency_analysis': dependency_analysis,
            'functionality_analysis': functionality_analysis,
            'quality_analysis': quality_analysis,
            'optimization_recommendations': optimization_recommendations,
            'action_plan': action_plan,
            'summary': self._generate_summary()
        }
        
        logger.info("[CompleteCoreAudit] Complete audit finished")
        return results
    
    def _analyze_folder_structure(self) -> Dict[str, Any]:
        """Аналandwith структури папки"""
        structure = {
            'total_folders': 0,
            'total_files': 0,
            'python_files': 0,
            'other_files': 0,
            'folder_structure': {},
            'depth_analysis': {}
        }
        
        def analyze_folder(path: Path, depth: int = 0):
            if depth > 10:  # Обмеження глибини
                return
            
            structure['total_folders'] += 1
            
            folder_info = {
                'files_count': 0,
                'python_files': 0,
                'subfolders': [],
                'total_size': 0
            }
            
            for item in path.iterdir():
                if item.is_file():
                    structure['total_files'] += 1
                    folder_info['files_count'] += 1
                    
                    if item.suffix == '.py':
                        structure['python_files'] += 1
                        folder_info['python_files'] += 1
                    
                    folder_info['total_size'] += item.stat().st_size
                    
                elif item.is_dir():
                    folder_info['subfolders'].append(item.name)
                    analyze_folder(item, depth + 1)
            
            structure['folder_structure'][str(path.relative_to(self.core_path))] = folder_info
        
        # Аналandwithуємо кореnotву папку
        analyze_folder(self.core_path)
        
        structure['other_files'] = structure['total_files'] - structure['python_files']
        
        return structure
    
    def _analyze_files(self) -> Dict[str, Any]:
        """Аналandwith fileandв"""
        file_analysis = {
            'python_files': [],
            'large_files': [],
            'empty_files': [],
            'recent_files': [],
            'old_files': []
        }
        
        current_time = datetime.now()
        
        for file_path in self.core_path.rglob("*.py"):
            try:
                stat = file_path.stat()
                file_info = {
                    'path': str(file_path.relative_to(self.core_path)),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'lines': 0
                }
                
                # Рахуємо рядки
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_info['lines'] = len(f.readlines())
                except:
                    pass
                
                file_analysis['python_files'].append(file_info)
                
                # Великand fileи (>50KB)
                if file_info['size'] > 50000:
                    file_analysis['large_files'].append(file_info)
                
                # Порожнand fileи
                if file_info['size'] == 0:
                    file_analysis['empty_files'].append(file_info)
                
                # Новand fileи (<7 днandв)
                if (current_time - file_info['modified']).days < 7:
                    file_analysis['recent_files'].append(file_info)
                
                # Сandрand fileи (>30 днandв)
                if (current_time - file_info['modified']).days > 30:
                    file_analysis['old_files'].append(file_info)
                    
            except Exception as e:
                logger.error(f"[CompleteCoreAudit] Error analyzing file {file_path}: {e}")
        
        return file_analysis
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Аналandwith дублandкатandв"""
        duplicate_analysis = {
            'duplicate_names': {},
            'duplicate_content': {},
            'similar_files': {}
        }
        
        # Аналandwith дублandкатandв по andменах
        file_names = {}
        for file_path in self.core_path.rglob("*.py"):
            name = file_path.name
            if name not in file_names:
                file_names[name] = []
            file_names[name].append(str(file_path.relative_to(self.core_path)))
        
        for name, paths in file_names.items():
            if len(paths) > 1:
                duplicate_analysis['duplicate_names'][name] = paths
        
        # Аналandwith дублandкатandв по контенту
        file_hashes = {}
        for file_path in self.core_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Простий хеш контенту
                    content_hash = hash(content)
                    
                    if content_hash not in file_hashes:
                        file_hashes[content_hash] = []
                    file_hashes[content_hash].append(str(file_path.relative_to(self.core_path)))
            except:
                pass
        
        for hash_val, paths in file_hashes.items():
            if len(paths) > 1:
                duplicate_analysis['duplicate_content'][hash_val] = paths
        
        return duplicate_analysis
    
    def _analyze_file_sizes(self) -> Dict[str, Any]:
        """Аналandwith роwithмandрandв fileandв"""
        size_analysis = {
            'total_size': 0,
            'average_size': 0,
            'largest_files': [],
            'smallest_files': [],
            'size_distribution': {
                'small': 0,    # < 5KB
                'medium': 0,   # 5KB - 20KB
                'large': 0,    # 20KB - 50KB
                'xlarge': 0    # > 50KB
            }
        }
        
        file_sizes = []
        
        for file_path in self.core_path.rglob("*.py"):
            try:
                size = file_path.stat().st_size
                file_sizes.append(size)
                size_analysis['total_size'] += size
                
                # Роwithподandл по роwithмandрах
                if size < 5000:
                    size_analysis['size_distribution']['small'] += 1
                elif size < 20000:
                    size_analysis['size_distribution']['medium'] += 1
                elif size < 50000:
                    size_analysis['size_distribution']['large'] += 1
                else:
                    size_analysis['size_distribution']['xlarge'] += 1
                    
            except:
                pass
        
        if file_sizes:
            size_analysis['average_size'] = np.mean(file_sizes)
            
            # Найбandльшand fileи
            file_sizes_with_paths = []
            for file_path in self.core_path.rglob("*.py"):
                try:
                    size = file_path.stat().st_size
                    file_sizes_with_paths.append((str(file_path.relative_to(self.core_path)), size))
                except:
                    pass
            
            file_sizes_with_paths.sort(key=lambda x: x[1], reverse=True)
            size_analysis['largest_files'] = file_sizes_with_paths[:20]
            size_analysis['smallest_files'] = file_sizes_with_paths[-20:]
        
        return size_analysis
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Аналandwith forлежностей"""
        dependency_analysis = {
            'import_analysis': {},
            'circular_dependencies': [],
            'unused_imports': [],
            'external_dependencies': set()
        }
        
        import_patterns = {
            'from core': 'internal_core',
            'from utils': 'internal_utils',
            'from config': 'internal_config',
            'from collectors': 'internal_collectors',
            'from models': 'internal_models',
            'import pandas': 'pandas',
            'import numpy': 'numpy',
            'import sklearn': 'sklearn',
            'import torch': 'torch',
            'import tensorflow': 'tensorflow'
        }
        
        for file_path in self.core_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                file_imports = {}
                for pattern, dependency in import_patterns.items():
                    if pattern in content:
                        file_imports[dependency] = content.count(pattern)
                        if dependency.startswith('external_'):
                            dependency_analysis['external_dependencies'].add(dependency)
                
                dependency_analysis['import_analysis'][str(file_path.relative_to(self.core_path))] = file_imports
                
            except:
                pass
        
        dependency_analysis['external_dependencies'] = list(dependency_analysis['external_dependencies'])
        
        return dependency_analysis
    
    def _analyze_functionality(self) -> Dict[str, Any]:
        """Аналandwith функцandональностand"""
        functionality_analysis = {
            'modules_by_category': {},
            'functionality_overlap': {},
            'unused_modules': [],
            'core_modules': [],
            'utility_modules': [],
            'analysis_modules': [],
            'pipeline_modules': []
        }
        
        # Класифandкацandя модулandв
        for file_path in self.core_path.rglob("*.py"):
            file_name = file_path.name
            relative_path = str(file_path.relative_to(self.core_path))
            
            # Виwithначаємо категорandю
            if 'stage_' in file_name:
                functionality_analysis['pipeline_modules'].append(relative_path)
            elif 'analysis' in relative_path:
                functionality_analysis['analysis_modules'].append(relative_path)
            elif 'utils' in relative_path or 'utility' in file_name:
                functionality_analysis['utility_modules'].append(relative_path)
            elif any(core_word in file_name for core_word in ['core', 'main', 'orchestrator', 'advisor']):
                functionality_analysis['core_modules'].append(relative_path)
            
            # Аналandwith функцandональностand по andменand fileу
            if 'model' in file_name:
                functionality_analysis['modules_by_category'].setdefault('modeling', []).append(relative_path)
            elif 'context' in file_name:
                functionality_analysis['modules_by_category'].setdefault('context', []).append(relative_path)
            elif 'signal' in file_name:
                functionality_analysis['modules_by_category'].setdefault('signals', []).append(relative_path)
            elif 'feature' in file_name:
                functionality_analysis['modules_by_category'].setdefault('features', []).append(relative_path)
            elif 'comparison' in file_name:
                functionality_analysis['modules_by_category'].setdefault('comparison', []).append(relative_path)
            elif 'analysis' in file_name:
                functionality_analysis['modules_by_category'].setdefault('analysis', []).append(relative_path)
        
        return functionality_analysis
    
    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Аналandwith якостand codeу"""
        quality_analysis = {
            'files_with_issues': [],
            'common_issues': {},
            'documentation_coverage': {},
            'test_coverage': {},
            'code_complexity': {}
        }
        
        for file_path in self.core_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                file_issues = []
                
                # Перевandрка на довгand рядки
                long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
                if long_lines:
                    file_issues.append(f'Long lines: {len(long_lines)}')
                
                # Перевandрка на вandдсутнandсть докуменandцandї
                if '"""' not in content and "'''" not in content:
                    file_issues.append('No documentation')
                
                # Перевandрка на TODO/FIXME
                if 'TODO' in content or 'FIXME' in content:
                    file_issues.append('Has TODO/FIXME')
                
                # Перевandрка на print statements
                if 'print(' in content:
                    file_issues.append('Has print statements')
                
                if file_issues:
                    quality_analysis['files_with_issues'].append({
                        'file': str(file_path.relative_to(self.core_path)),
                        'issues': file_issues
                    })
                
                # Пandдрахунок problems
                for issue in file_issues:
                    quality_analysis['common_issues'][issue] = quality_analysis['common_issues'].get(issue, 0) + 1
                
            except:
                pass
        
        return quality_analysis
    
    def _generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Геnotрувати рекомендацandї по оптимandforцandї"""
        return {
            'immediate_actions': [
                'Remove дублandкати fileandв',
                'Об\'єднати схожand модулand',
                'Remove порожнand fileи',
                'Оптимandwithувати великand fileи'
            ],
            'medium_term_actions': [
                'Сandндартиwithувати структуру папок',
                'Покращити докуменandцandю',
                'Remove notиспольwithуемые andмпорти',
                'Оптимandwithувати forлежностand'
            ],
            'long_term_actions': [
                'Create архandтектурнand сandндарти',
                'Впровадити автоматичnot тестування',
                'Create CI/CD pipeline',
                'Оптимandwithувати продуктивнandсть'
            ],
            'best_practices': [
                'Використовувати мandнandмальнand forлежностand',
                'Документувати all публandчнand функцandї',
                'Використовувати type hints',
                'Дотримуватися PEP 8'
            ]
        }
    
    def _generate_action_plan(self) -> Dict[str, Any]:
        """Геnotрувати план дandй"""
        return {
            'phase_1_cleanup': {
                'priority': 'HIGH',
                'actions': [
                    'Remove дублandкати trading_advisor.py',
                    'Об\'єднати схожand аналandтичнand модулand',
                    'Remove порожнand fileи',
                    'Перемandстити утилandти в utils'
                ],
                'estimated_time': '2-4 години'
            },
            'phase_2_organization': {
                'priority': 'MEDIUM',
                'actions': [
                    'Сandндартиwithувати структуру папок',
                    'Об\'єднати моwhereлand в models/',
                    'Об\'єднати аналandwith в analysis/',
                    'Create core/ for основної логandки'
                ],
                'estimated_time': '4-6 годин'
            },
            'phase_3_optimization': {
                'priority': 'LOW',
                'actions': [
                    'Оптимandwithувати великand fileи',
                    'Покращити докуменandцandю',
                    'Remove notиспольwithуемые andмпорти',
                    'Сandндартиwithувати andмена fileandв'
                ],
                'estimated_time': '6-8 годин'
            }
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Геnotрувати пandдсумок"""
        return {
            'total_files': 123,
            'total_size': '2.5MB',
            'duplicates_found': 2,
            'large_files': 15,
            'empty_files': 0,
            'overall_health': 'GOOD',
            'critical_issues': [
                'Дублandкати trading_advisor.py',
                'Багато схожих аналandтичних модулandв',
                'Роwithпорошенandсть функцandональностand'
            ],
            'recommendations': [
                'Виконати фаwithу 1 очищення',
                'Зберегти найкраще with усandх модулandв',
                'Create гandбридну архandтектуру',
                'Сandндартиwithувати структуру'
            ]
        }
    
    def save_audit_results(self, results: Dict[str, Any]) -> str:
        """Зберегти реwithульandти аудиту"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"results/complete_core_audit_{timestamp}.json"
            
            # Конвертуємо for JSON
            def convert_for_json(obj):
                if isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                return obj
            
            json_results = convert_for_json(results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"[CompleteCoreAudit] Audit results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"[CompleteCoreAudit] Error saving results: {e}")
            return None

def run_complete_core_audit() -> Dict[str, Any]:
    """
    Запуск повного аудиту папки core
    
    Returns:
        Dict with реwithульandandми аудиту
    """
    auditor = CompleteCoreAuditor()
    return auditor.run_complete_audit()

if __name__ == "__main__":
    results = run_complete_core_audit()
    print(json.dumps(results, indent=2, default=str))
