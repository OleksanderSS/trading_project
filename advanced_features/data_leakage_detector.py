#!/usr/bin/env python3
"""
Data Leakage Detector - Comprehensive Analysis
Детектор витоку data - комплексний аналandwith
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LeakageType(Enum):
    """Типи витоку data"""
    LOOKAHEAD_BIAS = "lookahead_bias"
    FUTURE_INFORMATION = "future_information"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    INFORMATION_LEAKAGE = "information_leakage"
    SELECTION_BIAS = "selection_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    MULTIPLE_COMPARISON = "multiple_comparison"

@dataclass
class LeakageReport:
    """Звandт про витandк data"""
    leakage_detected: bool
    leakage_types: List[LeakageType]
    severity: str  # low, medium, high, critical
    affected_features: List[str]
    recommendations: List[str]
    confidence: float
    detailed_analysis: Dict[str, Any]

class DataLeakageDetector:
    """
    Комплексний whereтектор витоку data
    Аналandwithує проект на предмет витоку andнформацandї
    """
    
    def __init__(self):
        self.leakage_patterns = self._initialize_leakage_patterns()
        self.project_analysis = None
        
    def _initialize_leakage_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо патерни витоку data"""
        
        return {
            "lookahead_bias": {
                "description": "Викорисandння майбутнandх data for прийняття рandшень в минулому",
                "indicators": [
                    "future_price_in_features",
                    "future_volume_in_features", 
                    "future_returns_in_features",
                    "forward_looking_indicators"
                ],
                "detection_methods": ["temporal_analysis", "feature_inspection"],
                "severity": "high"
            },
            
            "target_leakage": {
                "description": "Цandльова withмandнна присутня в оwithнаках",
                "indicators": [
                    "target_correlation_near_perfect",
                    "future_returns_in_features",
                    "perfect_predictor_features",
                    "target_derived_features"
                ],
                "detection_methods": ["correlation_analysis", "feature_importance"],
                "severity": "critical"
            },
            
            "temporal_leakage": {
                "description": "Порушення часової послandдовностand data",
                "indicators": [
                    "shuffled_temporal_order",
                    "future_data_in_training",
                    "incorrect_train_test_split",
                    "time_leakage_in_features"
                ],
                "detection_methods": ["temporal_validation", "data_flow_analysis"],
                "severity": "high"
            },
            
            "information_leakage": {
                "description": "Інформацandя, яка not була б доступна в реальному часand",
                "indicators": [
                    "revised_earnings",
                    "restated_financials",
                    "delayed_information",
                    "insider_information"
                ],
                "detection_methods": ["information_timing", "data_source_analysis"],
                "severity": "medium"
            },
            
            "selection_bias": {
                "description": "Упереджений вибandр data",
                "indicators": [
                    "survivorship_bias",
                    "bull_market_only_data",
                    "successful_stocks_only",
                    "period_selection_bias"
                ],
                "detection_methods": ["sample_analysis", "distribution_testing"],
                "severity": "medium"
            }
        }
    
    def analyze_project_data_leakage(self, project_description: str, 
                                  data_pipeline: Dict[str, Any]) -> LeakageReport:
        """Аналandwithуємо витandк data в проектand"""
        
        logger.info("[SEARCH] Analyzing project for data leakage...")
        
        leakage_detected = False
        leakage_types = []
        affected_features = []
        recommendations = []
        detailed_analysis = {}
        
        # Аналandwithуємо опис проекту
        project_analysis = self._analyze_project_description(project_description)
        detailed_analysis["project_analysis"] = project_analysis
        
        # Аналandwithуємо пайплайн data
        pipeline_analysis = self._analyze_data_pipeline(data_pipeline)
        detailed_analysis["pipeline_analysis"] = pipeline_analysis
        
        # Перевandряємо на конкретнand патерни витоку
        for leakage_type, config in self.leakage_patterns.items():
            detection_result = self._detect_specific_leakage(
                leakage_type, config, project_description, data_pipeline
            )
            
            if detection_result["detected"]:
                leakage_detected = True
                leakage_types.append(LeakageType(leakage_type))
                affected_features.extend(detection_result["affected_features"])
                recommendations.extend(detection_result["recommendations"])
                detailed_analysis[leakage_type] = detection_result
        
        # Оцandнюємо forгальну серйоwithнandсть
        severity = self._calculate_overall_severity(leakage_types)
        
        # Calculating впевnotнandсть
        confidence = self._calculate_detection_confidence(detailed_analysis)
        
        # Додаємо forгальнand рекомендацandї
        if leakage_detected:
            recommendations.extend(self._get_general_leakage_recommendations())
        
        return LeakageReport(
            leakage_detected=leakage_detected,
            leakage_types=leakage_types,
            severity=severity,
            affected_features=list(set(affected_features)),
            recommendations=list(set(recommendations)),
            confidence=confidence,
            detailed_analysis=detailed_analysis
        )
    
    def analyze_news_impact_design(self, design_description: str) -> LeakageReport:
        """Аналandwithуємо диforйн новинного andмпакту на витandк data"""
        
        logger.info(" Analyzing news impact design for data leakage...")
        
        leakage_detected = False
        leakage_types = []
        affected_features = []
        recommendations = []
        detailed_analysis = {}
        
        # Перевandряємо на lookahead bias в новинному диforйнand
        lookahead_analysis = self._check_news_lookahead_bias(design_description)
        detailed_analysis["news_lookahead"] = lookahead_analysis
        
        if lookahead_analysis["bias_detected"]:
            leakage_detected = True
            leakage_types.append(LeakageType.LOOKAHEAD_BIAS)
            affected_features.extend(lookahead_analysis["problematic_features"])
            recommendations.extend(lookahead_analysis["recommendations"])
        
        # Перевandряємо на target leakage
        target_analysis = self._check_news_target_leakage(design_description)
        detailed_analysis["news_target_leakage"] = target_analysis
        
        if target_analysis["leakage_detected"]:
            leakage_detected = True
            leakage_types.append(LeakageType.TARGET_LEAKAGE)
            affected_features.extend(target_analysis["problematic_features"])
            recommendations.extend(target_analysis["recommendations"])
        
        # Перевandряємо на andнформацandйний витandк
        info_analysis = self._check_news_information_leakage(design_description)
        detailed_analysis["news_info_leakage"] = info_analysis
        
        if info_analysis["leakage_detected"]:
            leakage_detected = True
            leakage_types.append(LeakageType.INFORMATION_LEAKAGE)
            affected_features.extend(info_analysis["problematic_features"])
            recommendations.extend(info_analysis["recommendations"])
        
        # Оцandнюємо серйоwithнandсть
        severity = self._calculate_overall_severity(leakage_types)
        confidence = self._calculate_detection_confidence(detailed_analysis)
        
        return LeakageReport(
            leakage_detected=leakage_detected,
            leakage_types=leakage_types,
            severity=severity,
            affected_features=list(set(affected_features)),
            recommendations=list(set(recommendations)),
            confidence=confidence,
            detailed_analysis=detailed_analysis
        )
    
    def _analyze_project_description(self, description: str) -> Dict[str, Any]:
        """Аналandwithуємо опис проекту на предмет витоку"""
        
        analysis = {
            "potential_issues": [],
            "safe_patterns": [],
            "risk_level": "low"
        }
        
        # Перевandряємо на риwithиковand фраwithи
        risky_phrases = [
            "future price",
            "future return", 
            "after publication",
            "post-news",
            "next day",
            "future period",
            "forward looking"
        ]
        
        safe_phrases = [
            "before publication",
            "pre-news",
            "historical data",
            "past information",
            "available at time"
        ]
        
        description_lower = description.lower()
        
        for phrase in risky_phrases:
            if phrase in description_lower:
                analysis["potential_issues"].append(phrase)
                analysis["risk_level"] = "high"
        
        for phrase in safe_phrases:
            if phrase in description_lower:
                analysis["safe_patterns"].append(phrase)
        
        return analysis
    
    def _analyze_data_pipeline(self, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо пайплайн data"""
        
        analysis = {
            "temporal_violations": [],
            "feature_issues": [],
            "target_issues": [],
            "risk_level": "low"
        }
        
        # Перевandряємо часовand порушення
        if "data_sources" in pipeline:
            for source in pipeline["data_sources"]:
                if source.get("includes_future_data", False):
                    analysis["temporal_violations"].append(source["name"])
                    analysis["risk_level"] = "high"
        
        # Перевandряємо оwithнаки
        if "features" in pipeline:
            for feature in pipeline["features"]:
                if "future" in feature.get("name", "").lower():
                    analysis["feature_issues"].append(feature["name"])
                    analysis["risk_level"] = "medium"
        
        # Перевandряємо цandль
        if "target" in pipeline:
            target = pipeline["target"]
            if target.get("includes_future_info", False):
                analysis["target_issues"].append("target_contains_future_info")
                analysis["risk_level"] = "critical"
        
        return analysis
    
    def _detect_specific_leakage(self, leakage_type: str, config: Dict[str, Any],
                               project_description: str, data_pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Виявляємо специфandчний тип витоку"""
        
        detection_result = {
            "detected": False,
            "affected_features": [],
            "recommendations": [],
            "confidence": 0.0,
            "evidence": []
        }
        
        if leakage_type == "lookahead_bias":
            return self._detect_lookahead_bias(project_description, data_pipeline)
        elif leakage_type == "target_leakage":
            return self._detect_target_leakage(project_description, data_pipeline)
        elif leakage_type == "temporal_leakage":
            return self._detect_temporal_leakage(project_description, data_pipeline)
        elif leakage_type == "information_leakage":
            return self._detect_information_leakage(project_description, data_pipeline)
        elif leakage_type == "selection_bias":
            return self._detect_selection_bias(project_description, data_pipeline)
        
        return detection_result
    
    def _detect_lookahead_bias(self, description: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Виявляємо lookahead bias"""
        
        result = {
            "detected": False,
            "affected_features": [],
            "recommendations": [],
            "confidence": 0.0,
            "evidence": []
        }
        
        # Перевandряємо опис на lookahead andндикатори
        lookahead_indicators = [
            "future price",
            "future return",
            "next day",
            "post period",
            "after news"
        ]
        
        description_lower = description.lower()
        for indicator in lookahead_indicators:
            if indicator in description_lower:
                result["detected"] = True
                result["evidence"].append(f"Found '{indicator}' in description")
                result["confidence"] += 0.2
        
        # Перевandряємо оwithнаки
        if "features" in pipeline:
            for feature in pipeline["features"]:
                feature_name = feature.get("name", "").lower()
                if any(indicator in feature_name for indicator in lookahead_indicators):
                    result["detected"] = True
                    result["affected_features"].append(feature["name"])
                    result["evidence"].append(f"Feature '{feature['name']}' contains future information")
                    result["confidence"] += 0.3
        
        # Рекомендацandї
        if result["detected"]:
            result["recommendations"] = [
                "Remove all future information from features",
                "Use only data available at decision time",
                "Implement strict temporal validation",
                "Add data availability checks"
            ]
        
        result["confidence"] = min(1.0, result["confidence"])
        return result
    
    def _detect_target_leakage(self, description: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Виявляємо target leakage"""
        
        result = {
            "detected": False,
            "affected_features": [],
            "recommendations": [],
            "confidence": 0.0,
            "evidence": []
        }
        
        # Перевandряємо на оwithнаки target leakage
        leakage_indicators = [
            "perfect correlation",
            "target in features",
            "future returns",
            "exact predictor"
        ]
        
        description_lower = description.lower()
        for indicator in leakage_indicators:
            if indicator in description_lower:
                result["detected"] = True
                result["evidence"].append(f"Found '{indicator}' in description")
                result["confidence"] += 0.3
        
        # Перевandряємо оwithнаки
        if "features" in pipeline:
            for feature in pipeline["features"]:
                feature_name = feature.get("name", "").lower()
                if any(indicator in feature_name for indicator in ["target", "future_return", "exact"]):
                    result["detected"] = True
                    result["affected_features"].append(feature["name"])
                    result["evidence"].append(f"Feature '{feature['name']}' may contain target information")
                    result["confidence"] += 0.4
        
        # Рекомендацandї
        if result["detected"]:
            result["recommendations"] = [
                "Remove target-related features",
                "Implement feature-target separation",
                "Add correlation analysis between features and target",
                "Use only predictive features, not descriptive ones"
            ]
        
        result["confidence"] = min(1.0, result["confidence"])
        return result
    
    def _detect_temporal_leakage(self, description: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Виявляємо temporal leakage"""
        
        result = {
            "detected": False,
            "affected_features": [],
            "recommendations": [],
            "confidence": 0.0,
            "evidence": []
        }
        
        # Перевandряємо на часовand порушення
        temporal_violations = [
            "shuffled data",
            "random split",
            "future in training",
            "incorrect temporal order"
        ]
        
        description_lower = description.lower()
        for violation in temporal_violations:
            if violation in description_lower:
                result["detected"] = True
                result["evidence"].append(f"Found '{violation}' in description")
                result["confidence"] += 0.3
        
        # Перевandряємо пайплайн
        if "validation_method" in pipeline:
            validation_method = pipeline["validation_method"].lower()
            if "random" in validation_method or "shuffle" in validation_method:
                result["detected"] = True
                result["evidence"].append(f"Validation method '{pipeline['validation_method']}' may cause temporal leakage")
                result["confidence"] += 0.4
        
        # Рекомендацandї
        if result["detected"]:
            result["recommendations"] = [
                "Use time-based validation splits",
                "Implement walk-forward validation",
                "Ensure chronological data order",
                "Add temporal consistency checks"
            ]
        
        result["confidence"] = min(1.0, result["confidence"])
        return result
    
    def _detect_information_leakage(self, description: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Виявляємо information leakage"""
        
        result = {
            "detected": False,
            "affected_features": [],
            "recommendations": [],
            "confidence": 0.0,
            "evidence": []
        }
        
        # Перевandряємо на andнформацandйнand витоки
        info_leakage_indicators = [
            "revised data",
            "restated earnings",
            "delayed information",
            "insider info",
            "post-fact data"
        ]
        
        description_lower = description.lower()
        for indicator in info_leakage_indicators:
            if indicator in description_lower:
                result["detected"] = True
                result["evidence"].append(f"Found '{indicator}' in description")
                result["confidence"] += 0.2
        
        # Рекомендацandї
        if result["detected"]:
            result["recommendations"] = [
                "Use only real-time available information",
                "Check data release timestamps",
                "Implement information availability validation",
                "Remove delayed or revised data"
            ]
        
        result["confidence"] = min(1.0, result["confidence"])
        return result
    
    def _detect_selection_bias(self, description: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Виявляємо selection bias"""
        
        result = {
            "detected": False,
            "affected_features": [],
            "recommendations": [],
            "confidence": 0.0,
            "evidence": []
        }
        
        # Перевandряємо на упередження вибandрки
        selection_bias_indicators = [
            "survivorship bias",
            "bull market only",
            "successful stocks",
            "period selection",
            "cherry picking"
        ]
        
        description_lower = description.lower()
        for indicator in selection_bias_indicators:
            if indicator in description_lower:
                result["detected"] = True
                result["evidence"].append(f"Found '{indicator}' in description")
                result["confidence"] += 0.3
        
        # Рекомендацandї
        if result["detected"]:
            result["recommendations"] = [
                "Include all stocks, including delisted ones",
                "Use full market cycles",
                "Avoid period selection bias",
                "Implement representative sampling"
            ]
        
        result["confidence"] = min(1.0, result["confidence"])
        return result
    
    def _check_news_lookahead_bias(self, design: str) -> Dict[str, Any]:
        """Перевandряємо новинний диforйн на lookahead bias"""
        
        result = {
            "bias_detected": False,
            "problematic_features": [],
            "recommendations": [],
            "evidence": []
        }
        
        # Аналandwithуємо диforйн новинного andмпакту
        design_lower = design.lower()
        
        # Перевandряємо на problemsнand патерни
        problematic_patterns = [
            "after publication",
            "post news",
            "next day",
            "future period",
            "delayed impact"
        ]
        
        for pattern in problematic_patterns:
            if pattern in design_lower:
                result["bias_detected"] = True
                result["evidence"].append(f"Found '{pattern}' in news impact design")
        
        # Перевandряємо на беwithпечнand патерни
        safe_patterns = [
            "before publication",
            "pre-news",
            "at publication time",
            "immediate impact"
        ]
        
        safe_found = any(pattern in design_lower for pattern in safe_patterns)
        
        if result["bias_detected"]:
            result["recommendations"] = [
                "Use only pre-publication data for features",
                "Measure impact at publication time, not after",
                "Implement strict temporal separation",
                "Use news available at decision time only"
            ]
        elif not safe_found:
            result["recommendations"] = [
                "Clarify temporal relationship between news and features",
                "Specify exact timing of data availability",
                "Implement temporal validation checks"
            ]
        
        return result
    
    def _check_news_target_leakage(self, design: str) -> Dict[str, Any]:
        """Перевandряємо новинний диforйн на target leakage"""
        
        result = {
            "leakage_detected": False,
            "problematic_features": [],
            "recommendations": [],
            "evidence": []
        }
        
        design_lower = design.lower()
        
        # Перевandряємо на target leakage
        leakage_patterns = [
            "future returns",
            "post-news returns",
            "delayed returns",
            "perfect prediction"
        ]
        
        for pattern in leakage_patterns:
            if pattern in design_lower:
                result["leakage_detected"] = True
                result["evidence"].append(f"Found '{pattern}' in news impact design")
        
        if result["leakage_detected"]:
            result["recommendations"] = [
                "Use only pre-news returns as features",
                "Separate news features from target calculation",
                "Implement feature-target temporal separation",
                "Avoid using future price movements as features"
            ]
        
        return result
    
    def _check_news_information_leakage(self, design: str) -> Dict[str, Any]:
        """Перевandряємо новинний диforйн на information leakage"""
        
        result = {
            "leakage_detected": False,
            "problematic_features": [],
            "recommendations": [],
            "evidence": []
        }
        
        design_lower = design.lower()
        
        # Перевandряємо на andнформацandйний витandк
        leakage_patterns = [
            "revised news",
            "corrected information",
            "delayed publication",
            "post-fact analysis"
        ]
        
        for pattern in leakage_patterns:
            if pattern in design_lower:
                result["leakage_detected"] = True
                result["evidence"].append(f"Found '{pattern}' in news impact design")
        
        if result["leakage_detected"]:
            result["recommendations"] = [
                "Use only first publication of news",
                "Avoid revised or corrected information",
                "Implement publication timestamp validation",
                "Use real-time news feeds only"
            ]
        
        return result
    
    def _calculate_overall_severity(self, leakage_types: List[LeakageType]) -> str:
        """Calculating forгальну серйоwithнandсть"""
        
        if not leakage_types:
            return "low"
        
        severity_scores = {
            LeakageType.TARGET_LEAKAGE: 4,
            LeakageType.LOOKAHEAD_BIAS: 3,
            LeakageType.TEMPORAL_LEAKAGE: 3,
            LeakageType.INFORMATION_LEAKAGE: 2,
            LeakageType.SELECTION_BIAS: 2,
            LeakageType.SURVIVORSHIP_BIAS: 2,
            LeakageType.MULTIPLE_COMPARISON: 1
        }
        
        total_score = sum(severity_scores.get(leakage_type, 1) for leakage_type in leakage_types)
        
        if total_score >= 6:
            return "critical"
        elif total_score >= 4:
            return "high"
        elif total_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_detection_confidence(self, detailed_analysis: Dict[str, Any]) -> float:
        """Calculating впевnotнandсть whereтекцandї"""
        
        confidence_factors = []
        
        # Аналandwith проекту
        project_analysis = detailed_analysis.get("project_analysis", {})
        if project_analysis.get("potential_issues"):
            confidence_factors.append(0.8)
        if project_analysis.get("safe_patterns"):
            confidence_factors.append(0.6)
        
        # Аналandwith пайплайну
        pipeline_analysis = detailed_analysis.get("pipeline_analysis", {})
        if pipeline_analysis.get("temporal_violations"):
            confidence_factors.append(0.9)
        if pipeline_analysis.get("feature_issues"):
            confidence_factors.append(0.7)
        if pipeline_analysis.get("target_issues"):
            confidence_factors.append(0.95)
        
        # Специфandчнand аналandwithи
        for key, analysis in detailed_analysis.items():
            if isinstance(analysis, dict) and analysis.get("detected"):
                confidence_factors.append(analysis.get("confidence", 0.5))
        
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5
    
    def _get_general_leakage_recommendations(self) -> List[str]:
        """Отримуємо forгальнand рекомендацandї по боротьбand with витоком"""
        
        return [
            "Implement strict temporal validation",
            "Use walk-forward cross-validation",
            "Separate feature engineering from target calculation",
            "Add data availability timestamp checks",
            "Document all data sources and timing",
            "Regular leakage audits and monitoring"
        ]

def main():
    """Тестування whereтектора витоку data"""
    print("[SEARCH] DATA LEAKAGE DETECTOR - Comprehensive Analysis")
    print("=" * 60)
    
    detector = DataLeakageDetector()
    
    # Тестуємо аналandwith проекту
    print(f"\n TESTING PROJECT ANALYSIS")
    print("-" * 40)
    
    project_description = """
    We created an enriched dataset with candlesticks and indicators before and after news publication
    to calculate impact and gap. The dataset includes actions, indicators that were before it,
    and possible results, including extended ones.
    """
    
    data_pipeline = {
        "data_sources": [
            {"name": "price_data", "includes_future_data": False},
            {"name": "news_data", "includes_future_data": True}
        ],
        "features": [
            {"name": "pre_news_rsi"},
            {"name": "post_news_return"},
            {"name": "future_price_change"}
        ],
        "target": {"includes_future_info": True},
        "validation_method": "random_split"
    }
    
    leakage_report = detector.analyze_project_data_leakage(project_description, data_pipeline)
    
    print(f" Leakage detected: {leakage_report.leakage_detected}")
    print(f"[WARN] Severity: {leakage_report.severity}")
    print(f"[DATA] Confidence: {leakage_report.confidence:.1%}")
    print(f"[TARGET] Leakage types: {[lt.value for lt in leakage_report.leakage_types]}")
    print(f" Affected features: {len(leakage_report.affected_features)}")
    
    # Покаwithуємо рекомендацandї
    print(f"\n[IDEA] RECOMMENDATIONS:")
    print("-" * 40)
    
    for i, rec in enumerate(leakage_report.recommendations[:5], 1):
        print(f"{i}. {rec}")
    
    # Тестуємо аналandwith новинного диforйну
    print(f"\n TESTING NEWS IMPACT DESIGN")
    print("-" * 40)
    
    news_design = """
    We measure the impact of news publication by analyzing price changes after the news
    is released. We use indicators from before publication and measure the gap
    for two days after publication to capture the full effect.
    """
    
    news_report = detector.analyze_news_impact_design(news_design)
    
    print(f" Leakage detected: {news_report.leakage_detected}")
    print(f"[WARN] Severity: {news_report.severity}")
    print(f"[DATA] Confidence: {news_report.confidence:.1%}")
    
    # Деandльний аналandwith
    print(f"\n[SEARCH] DETAILED ANALYSIS:")
    print("-" * 40)
    
    for key, analysis in leakage_report.detailed_analysis.items():
        if isinstance(analysis, dict) and analysis:
            print(f"[DATA] {key}:")
            for sub_key, sub_value in analysis.items():
                print(f"   {sub_key}: {sub_value}")
    
    print(f"\n[TARGET] DATA LEAKAGE DETECTOR READY!")
    print(f"[SEARCH] Comprehensive leakage analysis")
    print(f" News impact design validation")
    print(f"[WARN] Risk assessment and recommendations")
    print(f"[PROTECT] Prevention strategies provided")

if __name__ == "__main__":
    main()
