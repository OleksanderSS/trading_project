#!/usr/bin/env python3
"""
Detailed System Audit Script
Checks all components, models, analyzers, and their status
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemAuditor:
    def __init__(self):
        self.root_dir = Path("d:/trading_project")
        self.data_dir = self.root_dir / "data"
        self.models_dir = self.data_dir / "colab" / "accumulated" / "main_database"
        self.results_dir = self.root_dir / "data" / "results"
        
    def audit_model_types(self) -> Dict[str, Any]:
        """Audit available model types and their counts"""
        logger.info("🔍 Auditing model types...")
        
        model_stats = {
            'mlp': {'files': [], 'count': 0},
            'tabnet': {'files': [], 'count': 0},
            'keras': {'files': [], 'count': 0},
            'other': {'files': [], 'count': 0}
        }
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return model_stats
            
        # Scan for model files
        for file_path in self.models_dir.glob("*"):
            if file_path.is_file():
                file_name = file_path.name.lower()
                
                if file_name.endswith('.pkl'):
                    model_stats['mlp']['files'].append(file_path.name)
                    model_stats['mlp']['count'] += 1
                elif file_name.endswith('.zip'):
                    model_stats['tabnet']['files'].append(file_path.name)
                    model_stats['tabnet']['count'] += 1
                elif file_name.endswith(('.keras', '.h5', '.pb')):
                    model_stats['keras']['files'].append(file_path.name)
                    model_stats['keras']['count'] += 1
                else:
                    model_stats['other']['files'].append(file_path.name)
                    model_stats['other']['count'] += 1
        
        return model_stats
    
    def audit_scalers(self) -> Dict[str, Any]:
        """Audit available target scalers"""
        logger.info("🔍 Auditing target scalers...")
        
        scaler_stats = {'scalers': [], 'count': 0}
        
        if not self.models_dir.exists():
            return scaler_stats
            
        # Look for scaler files
        for file_path in self.models_dir.glob("scaler_*.pkl"):
            scaler_stats['scalers'].append(file_path.name)
            scaler_stats['count'] += 1
            
        return scaler_stats
    
    def audit_pipeline_components(self) -> Dict[str, Any]:
        """Audit pipeline stages and their components"""
        logger.info("🔍 Auditing pipeline components...")
        
        components = {
            'stages': {},
            'analyzers': {},
            'engines': {},
            'calculators': {}
        }
        
        # Check pipeline stages
        stages_dir = self.root_dir / "src" / "pipeline" / "stages"
        if stages_dir.exists():
            for stage_file in stages_dir.glob("stage_*.py"):
                stage_name = stage_file.stem
                components['stages'][stage_name] = {
                    'file': str(stage_file),
                    'size': stage_file.stat().st_size,
                    'exists': True
                }
        
        # Check analyzers
        analyzers_dir = self.root_dir / "src" / "analytics" / "analyzers"
        if analyzers_dir.exists():
            for analyzer_file in analyzers_dir.glob("*_analyzer.py"):
                analyzer_name = analyzer_file.stem
                components['analyzers'][analyzer_name] = {
                    'file': str(analyzer_file),
                    'size': analyzer_file.stat().st_size,
                    'exists': True
                }
        
        # Check engines
        engines_dir = self.root_dir / "src" / "analytics"
        if engines_dir.exists():
            for engine_file in engines_dir.glob("*_engine.py"):
                engine_name = engine_file.stem
                components['engines'][engine_name] = {
                    'file': str(engine_file),
                    'size': engine_file.stat().st_size,
                    'exists': True
                }
        
        # Check calculators
        calculators_dir = self.root_dir / "src" / "analytics" / "calculators"
        if calculators_dir.exists():
            for calc_file in calculators_dir.glob("*_calculator.py"):
                calc_name = calc_file.stem
                components['calculators'][calc_name] = {
                    'file': str(calc_file),
                    'size': calc_file.stat().st_size,
                    'exists': True
                }
        
        return components
    
    def audit_recent_results(self) -> Dict[str, Any]:
        """Audit recent pipeline results"""
        logger.info("🔍 Auditing recent results...")
        
        results = {
            'latest_results': [],
            'evaluation_summaries': [],
            'backtest_results': []
        }
        
        if not self.results_dir.exists():
            return results
            
        # Get latest result files
        result_files = list(self.results_dir.glob("*.json"))
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for result_file in result_files[:10]:  # Last 10 results
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                file_info = {
                    'file': result_file.name,
                    'size': result_file.stat().st_size,
                    'modified': result_file.stat().st_mtime,
                    'keys': list(data.keys()) if isinstance(data, dict) else [],
                    'has_evaluation': 'evaluation_summary' in str(data),
                    'has_backtest': 'backtest' in str(data).lower(),
                    'has_performance': 'performance' in str(data).lower()
                }
                
                if 'evaluation' in result_file.name.lower():
                    results['evaluation_summaries'].append(file_info)
                elif 'backtest' in result_file.name.lower():
                    results['backtest_results'].append(file_info)
                else:
                    results['latest_results'].append(file_info)
                    
            except Exception as e:
                logger.warning(f"Could not read {result_file}: {e}")
        
        return results
    
    def audit_data_accumulation(self) -> Dict[str, Any]:
        """Audit data accumulation status"""
        logger.info("🔍 Auditing data accumulation...")
        
        data_stats = {
            'accumulated_batches': [],
            'total_size_mb': 0,
            'file_count': 0,
            'latest_batch': None
        }
        
        accumulated_dir = self.data_dir / "colab" / "accumulated"
        if not accumulated_dir.exists():
            return data_stats
            
        for batch_dir in accumulated_dir.iterdir():
            if batch_dir.is_dir():
                batch_info = {
                    'name': batch_dir.name,
                    'file_count': len(list(batch_dir.glob("*"))),
                    'size_mb': sum(f.stat().st_size for f in batch_dir.glob("*")) / (1024*1024)
                }
                data_stats['accumulated_batches'].append(batch_info)
                data_stats['total_size_mb'] += batch_info['size_mb']
                data_stats['file_count'] += batch_info['file_count']
                
                if data_stats['latest_batch'] is None or batch_dir.stat().st_mtime > data_stats['latest_batch']['modified']:
                    data_stats['latest_batch'] = {
                        'name': batch_dir.name,
                        'modified': batch_dir.stat().st_mtime
                    }
        
        return data_stats
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system audit report"""
        logger.info("📊 Generating comprehensive system audit...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'system_status': 'healthy',
            'warnings': [],
            'recommendations': [],
            'components': {
                'models': self.audit_model_types(),
                'scalers': self.audit_scalers(),
                'pipeline': self.audit_pipeline_components(),
                'results': self.audit_recent_results(),
                'data_accumulation': self.audit_data_accumulation()
            }
        }
        
        # Analyze and add warnings/recommendations
        model_stats = report['components']['models']
        
        if model_stats['mlp']['count'] == 0:
            report['warnings'].append("No MLP models found")
            report['system_status'] = 'degraded'
            
        if model_stats['tabnet']['count'] == 0:
            report['warnings'].append("No TabNet models found")
            
        if model_stats['keras']['count'] == 0:
            report['warnings'].append("No Keras models found (CNN, LSTM, etc.)")
            report['recommendations'].append("Consider training deep learning models for better performance")
            
        scaler_stats = report['components']['scalers']
        if scaler_stats['count'] < model_stats['mlp']['count']:
            report['warnings'].append(f"Missing scalers: {model_stats['mlp']['count'] - scaler_stats['count']} scalers missing")
            report['recommendations'].append("Ensure target scalers are saved during model training")
            
        # Check pipeline components
        pipeline_stats = report['components']['pipeline']
        if len(pipeline_stats['stages']) < 7:
            report['warnings'].append(f"Missing pipeline stages: found {len(pipeline_stats['stages'])}, expected 7")
            
        if len(pipeline_stats['analyzers']) < 3:
            report['warnings'].append(f"Limited analyzers: found {len(pipeline_stats['analyzers'])}")
            
        # Check recent results
        results_stats = report['components']['results']
        if len(results_stats['evaluation_summaries']) == 0:
            report['warnings'].append("No recent evaluation summaries found")
            report['recommendations'].append("Run pipeline with evaluation stage to generate summaries")
            
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save audit report to file"""
        if filename is None:
            filename = f"system_audit_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        report_file = self.root_dir / "reports" / filename
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"📄 Audit report saved to: {report_file}")
        return report_file

def main():
    """Main audit execution"""
    logger.info("🚀 Starting comprehensive system audit...")
    
    auditor = SystemAuditor()
    report = auditor.generate_comprehensive_report()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 SYSTEM AUDIT SUMMARY")
    print("="*60)
    print(f"Status: {report['system_status'].upper()}")
    print(f"Models Found: {report['components']['models']['mlp']['count']} MLP, {report['components']['models']['tabnet']['count']} TabNet")
    print(f"Scalers Found: {report['components']['scalers']['count']}")
    print(f"Pipeline Stages: {len(report['components']['pipeline']['stages'])}")
    print(f"Analyzers: {len(report['components']['pipeline']['analyzers'])}")
    print(f"Recent Results: {len(report['components']['results']['latest_results'])}")
    
    if report['warnings']:
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"   • {warning}")
    
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS ({len(report['recommendations'])}):")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    print("="*60)
    
    # Save detailed report
    auditor.save_report(report)
    
    return report

if __name__ == "__main__":
    main()
