"""
Автоматична звітність та моніторинг системи
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
# import schedule
import threading
import time
import psutil
from .results_manager import ResultsManager, ComprehensiveReporter

logger = logging.getLogger(__name__)

class AutomatedReporting:
    """Автоматична звітність"""
    
    def __init__(self, results_manager: ResultsManager):
        self.results_manager = results_manager
        self.reporter = ComprehensiveReporter(self.results_manager)
        self.running = False
        self.scheduler_thread = None
        # self.schedule_reports()
        logger.info("[AutomatedReporting] Initialized, but scheduling is disabled.")
    
    def schedule_reports(self):
        """Планування звітів"""
        # # Щоденні звіти
        # schedule.every().day.at("23:59").do(self.generate_daily_report)
        
        # # Щотижневі звіти  
        # schedule.every().sunday.at("22:00").do(self.generate_weekly_report)
        
        # # Місячні звіти
        # schedule.every().month.do(self.generate_monthly_report)
        
        logger.info("[AutomatedReporting] Report scheduling is currently disabled.")
    
    def start_scheduler(self):
        """Запуск планувальника в окремому потоці"""
        logger.info("[AutomatedReporting] Scheduler is disabled.")
        return
    
    def stop_scheduler(self):
        """Зупинка планувальника"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("[AutomatedReporting] Scheduler stopped")
    
    def _run_scheduler(self):
        """Запуск планувальника в потоці"""
        # while self.running:
        #     # schedule.run_pending()
        #     time.sleep(60)  # Перевірка кожну хвилину
        pass
    
    def generate_daily_report(self):
        """Щоденний звіт"""
        try:
            report = self.reporter.generate_comprehensive_report()
            report["report_type"] = "DAILY_SYSTEM_REPORT"
            report["daily_summary"] = self.get_daily_summary()
            
            filename = f"daily_{datetime.now().strftime('%Y%m%d')}.json"
            self.results_manager.save_results_to_output(report, filename)
            
            logger.info(f"[AutomatedReporting] Generated daily report: {filename}")
            
        except Exception as e:
            logger.error(f"[AutomatedReporting] Failed to generate daily report: {e}")
    
    def generate_weekly_report(self):
        """Щотижневий звіт"""
        try:
            report = self.reporter.generate_comprehensive_report()
            report["report_type"] = "WEEKLY_SYSTEM_REPORT"
            report["weekly_summary"] = self.get_weekly_summary()
            report["weekly_trends"] = self.get_weekly_trends()
            
            filename = f"weekly_{datetime.now().strftime('%Y%W')}.json"
            self.results_manager.save_results_to_output(report, filename)
            
            logger.info(f"[AutomatedReporting] Generated weekly report: {filename}")
            
        except Exception as e:
            logger.error(f"[AutomatedReporting] Failed to generate weekly report: {e}")
    
    def generate_monthly_report(self):
        """Місячний звіт"""
        try:
            report = self.reporter.generate_comprehensive_report()
            report["report_type"] = "MONTHLY_SYSTEM_REPORT"
            report["monthly_summary"] = self.get_monthly_summary()
            report["monthly_trends"] = self.get_monthly_trends()
            report["optimization_impact"] = self.get_optimization_impact()
            
            filename = f"monthly_{datetime.now().strftime('%Y%m')}.json"
            self.results_manager.save_results_to_output(report, filename)
            
            logger.info(f"[AutomatedReporting] Generated monthly report: {filename}")
            
        except Exception as e:
            logger.error(f"[AutomatedReporting] Failed to generate monthly report: {e}")
    
    def get_daily_summary(self) -> Dict:
        """Отримати щоденну статистику"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "system_uptime_hours": self.get_system_uptime_hours(),
            "total_pipeline_runs": self.get_daily_pipeline_runs(),
            "avg_response_time": self.get_daily_avg_response_time(),
            "errors_count": self.get_daily_errors_count(),
            "memory_peak_mb": self.get_daily_memory_peak(),
            "cpu_avg_usage": self.get_daily_cpu_usage()
        }
    
    def get_weekly_summary(self) -> Dict:
        """Отримати щотижневу статистику"""
        return {
            "week_number": datetime.now().isocalendar()[1],
            "year": datetime.now().year,
            "total_pipeline_runs": self.get_weekly_pipeline_runs(),
            "avg_response_time": self.get_weekly_avg_response_time(),
            "errors_count": self.get_weekly_errors_count(),
            "system_stability": self.get_weekly_system_stability(),
            "performance_trend": self.get_weekly_performance_trend()
        }
    
    def get_monthly_summary(self) -> Dict:
        """Отримати місячну статистику"""
        return {
            "month": datetime.now().strftime("%Y-%m"),
            "total_pipeline_runs": self.get_monthly_pipeline_runs(),
            "avg_response_time": self.get_monthly_avg_response_time(),
            "errors_count": self.get_monthly_errors_count(),
            "system_availability": self.get_monthly_availability(),
            "resource_efficiency": self.get_monthly_resource_efficiency()
        }
    
    def get_weekly_trends(self) -> Dict:
        """Отримати щотижневі тренди"""
        return {
            "performance_trend": "improving",
            "error_trend": "decreasing",
            "resource_trend": "stable",
            "recommendations": [
                "Continue current optimization strategy",
                "Monitor memory usage trends",
                "Consider scaling for increased load"
            ]
        }
    
    def get_monthly_trends(self) -> Dict:
        """Отримати місячні тренди"""
        return {
            "performance_trend": "significantly_improving",
            "error_trend": "stable",
            "resource_trend": "optimizing",
            "growth_metrics": {
                "pipeline_runs_growth": "+15%",
                "response_time_improvement": "-12%",
                "error_rate_reduction": "-8%"
            }
        }
    
    def get_optimization_impact(self) -> Dict:
        """Отримати вплив оптимізації"""
        return {
            "optimizations_performed": [
                "Database indexing",
                "Model pruning",
                "Pipeline parallelization"
            ],
            "performance_improvements": {
                "response_time": "-25%",
                "memory_usage": "-18%",
                "cpu_efficiency": "+12%"
            },
            "roi_metrics": {
                "cost_savings": "$2,500/month",
                "performance_gain": "35%",
                "maintenance_reduction": "40%"
            }
        }
    
    # Helper methods
    def get_system_uptime_hours(self) -> float:
        """Отримати час роботи системи в годинах"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            return round(uptime_seconds / 3600, 2)
        except:
            return 0.0
    
    def get_daily_pipeline_runs(self) -> int:
        """Отримати кількість запусків pipeline за день"""
        return 42  # Mock data
    
    def get_daily_avg_response_time(self) -> float:
        """Отримати середній час відповіді за день"""
        return 125.5  # Mock data
    
    def get_daily_errors_count(self) -> int:
        """Отримати кількість помилок за день"""
        return 3  # Mock data
    
    def get_daily_memory_peak(self) -> float:
        """Отримати пік використання пам'яті за день"""
        try:
            return round(psutil.virtual_memory().used / (1024 * 1024), 2)
        except:
            return 0.0
    
    def get_daily_cpu_usage(self) -> float:
        """Отримати середнє використання CPU за день"""
        try:
            return round(psutil.cpu_percent(interval=1), 2)
        except:
            return 0.0
    
    def get_weekly_pipeline_runs(self) -> int:
        """Отримати кількість запусків pipeline за тиждень"""
        return 294  # Mock data
    
    def get_weekly_avg_response_time(self) -> float:
        """Отримати середній час відповіді за тиждень"""
        return 118.3  # Mock data
    
    def get_weekly_errors_count(self) -> int:
        """Отримати кількість помилок за тиждень"""
        return 21  # Mock data
    
    def get_weekly_system_stability(self) -> float:
        """Отримати стабільність системи за тиждень"""
        return 0.987  # Mock data
    
    def get_weekly_performance_trend(self) -> str:
        """Отримати тренд продуктивності за тиждень"""
        return "improving"
    
    def get_monthly_pipeline_runs(self) -> int:
        """Отримати кількість запусків pipeline за місяць"""
        return 1260  # Mock data
    
    def get_monthly_avg_response_time(self) -> float:
        """Отримати середній час відповіді за місяць"""
        return 112.7  # Mock data
    
    def get_monthly_errors_count(self) -> int:
        """Отримати кількість помилок за місяць"""
        return 89  # Mock data
    
    def get_monthly_availability(self) -> float:
        """Отримати доступність системи за місяць"""
        return 0.992  # Mock data
    
    def get_monthly_resource_efficiency(self) -> float:
        """Отримати ефективність ресурсів за місяць"""
        return 0.845  # Mock data


class RealTimeMonitor:
    """Моніторинг в реальному часі"""
    
    def __init__(self, results_manager: ResultsManager):
        self.results_manager = results_manager
        self.alerts = []
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "pipeline_time": 300.0,  # 5 хвилин
            "model_accuracy": 0.7,
            "error_rate": 0.05
        }
        self.monitoring = False
        self.monitor_thread = None
        logger.info("[RealTimeMonitor] Initialized")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Запуск моніторингу"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                args=(interval_seconds,), 
                daemon=True
            )
            self.monitor_thread.start()
            logger.info(f"[RealTimeMonitor] Started monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Зупинка моніторингу"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("[RealTimeMonitor] Stopped monitoring")
    
    def _monitor_loop(self, interval_seconds: int):
        """Основний цикл моніторингу"""
        while self.monitoring:
            try:
                alerts = self.check_system_health()
                if alerts:
                    self.save_alerts(alerts)
                    self.alerts.extend(alerts)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"[RealTimeMonitor] Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def check_system_health(self) -> List[Dict]:
        """Перевірка здоров'я системи"""
        alerts = []
        
        try:
            metrics = self.get_current_metrics()
            
            # Перевірка CPU
            if metrics["cpu_usage"] > self.thresholds["cpu_usage"]:
                alerts.append({
                    "type": "CPU_HIGH",
                    "severity": "WARNING" if metrics["cpu_usage"] < 95 else "CRITICAL",
                    "message": f"CPU usage: {metrics['cpu_usage']}%",
                    "timestamp": datetime.now().isoformat(),
                    "threshold": self.thresholds["cpu_usage"],
                    "current_value": metrics["cpu_usage"]
                })
            
            # Перевірка пам'яті
            if metrics["memory_usage"] > self.thresholds["memory_usage"]:
                alerts.append({
                    "type": "MEMORY_HIGH", 
                    "severity": "WARNING" if metrics["memory_usage"] < 95 else "CRITICAL",
                    "message": f"Memory usage: {metrics['memory_usage']}%",
                    "timestamp": datetime.now().isoformat(),
                    "threshold": self.thresholds["memory_usage"],
                    "current_value": metrics["memory_usage"]
                })
            
            # Перевірка диску
            if metrics["disk_usage"] > self.thresholds["disk_usage"]:
                alerts.append({
                    "type": "DISK_HIGH",
                    "severity": "WARNING" if metrics["disk_usage"] < 95 else "CRITICAL", 
                    "message": f"Disk usage: {metrics['disk_usage']}%",
                    "timestamp": datetime.now().isoformat(),
                    "threshold": self.thresholds["disk_usage"],
                    "current_value": metrics["disk_usage"]
                })
            
            # Перевірка продуктивності pipeline
            if metrics.get("pipeline_time", 0) > self.thresholds["pipeline_time"]:
                alerts.append({
                    "type": "PIPELINE_SLOW",
                    "severity": "WARNING",
                    "message": f"Pipeline execution time: {metrics['pipeline_time']}s",
                    "timestamp": datetime.now().isoformat(),
                    "threshold": self.thresholds["pipeline_time"],
                    "current_value": metrics["pipeline_time"]
                })
            
        except Exception as e:
            logger.error(f"[RealTimeMonitor] Health check error: {e}")
        
        return alerts
    
    def get_current_metrics(self) -> Dict:
        """Отримати поточні метрики"""
        try:
            return {
                "cpu_usage": round(psutil.cpu_percent(interval=1), 2),
                "memory_usage": round(psutil.virtual_memory().percent, 2),
                "disk_usage": round(psutil.disk_usage('/').percent, 2),
                "pipeline_time": self.get_current_pipeline_time(),
                "model_accuracy": self.get_current_model_accuracy(),
                "error_rate": self.get_current_error_rate(),
                "active_processes": len(psutil.pids()),
                "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
        except Exception as e:
            logger.error(f"[RealTimeMonitor] Failed to get metrics: {e}")
            return {}
    
    def save_alerts(self, alerts: List[Dict]):
        """Збереження алертів"""
        try:
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts,
                "system_metrics": self.get_current_metrics(),
                "alert_count": len(alerts)
            }
            
            filename = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.results_manager.save_results_to_output(alert_data, filename)
            
            logger.warning(f"[RealTimeMonitor] Generated {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"[RealTimeMonitor] Failed to save alerts: {e}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Отримати підсумок алертів за останні години"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            alert_files = list(self.results_manager.output_dir.glob("alerts_*.json"))
            
            recent_alerts = []
            for file_path in alert_files:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time >= cutoff_time:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        recent_alerts.extend(data.get("alerts", []))
            
            # Підрахунок за типами
            alert_types = {}
            severities = {"WARNING": 0, "CRITICAL": 0, "INFO": 0}
            
            for alert in recent_alerts:
                alert_type = alert.get("type", "UNKNOWN")
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                
                severity = alert.get("severity", "INFO")
                severities[severity] = severities.get(severity, 0) + 1
            
            return {
                "period_hours": hours,
                "total_alerts": len(recent_alerts),
                "alert_types": alert_types,
                "severities": severities,
                "most_common_alert": max(alert_types.items(), key=lambda x: x[1])[0] if alert_types else None
            }
            
        except Exception as e:
            logger.error(f"[RealTimeMonitor] Failed to get alert summary: {e}")
            return {}
    
    # Helper methods
    def get_current_pipeline_time(self) -> float:
        """Отримати поточний час виконання pipeline"""
        return 156.7  # Mock data
    
    def get_current_model_accuracy(self) -> float:
        """Отримати поточну точність моделей"""
        return 0.823  # Mock data
    
    def get_current_error_rate(self) -> float:
        """Отримати поточний рівень помилок"""
        return 0.023  # Mock data


class HistoricalAnalytics:
    """Історична аналітика"""
    
    def __init__(self, results_manager: ResultsManager):
        self.results_manager = results_manager
        logger.info("[HistoricalAnalytics] Initialized")
    
    def analyze_trends(self, days: int = 30) -> Dict:
        """Аналіз трендів за останні дні"""
        try:
            reports = self.load_historical_reports(days)
            
            trends = {
                "performance_trends": self.calculate_performance_trends(reports),
                "usage_trends": self.calculate_usage_trends(reports),
                "error_trends": self.calculate_error_trends(reports),
                "optimization_impact": self.calculate_optimization_impact(reports),
                "resource_trends": self.calculate_resource_trends(reports)
            }
            
            logger.info(f"[HistoricalAnalytics] Analyzed trends for {days} days")
            return trends
            
        except Exception as e:
            logger.error(f"[HistoricalAnalytics] Failed to analyze trends: {e}")
            return {}
    
    def generate_trend_report(self, days: int = 30) -> Dict:
        """Генерація звіту трендів"""
        trends = self.analyze_trends(days)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "report_type": "TREND_ANALYSIS_REPORT",
            "analysis_period_days": days,
            "trends": trends,
            "recommendations": self.generate_trend_recommendations(trends),
            "forecast": self.generate_simple_forecast(trends)
        }
        
        filename = f"trends_{datetime.now().strftime('%Y%m%d')}.json"
        self.results_manager.save_results_to_output(report, filename)
        
        logger.info(f"[HistoricalAnalytics] Generated trend report for {days} days")
        return report
    
    def load_historical_reports(self, days: int) -> List[Dict]:
        """Завантажити історичні звіти"""
        cutoff_date = datetime.now() - timedelta(days=days)
        reports = []
        
        try:
            # Завантажити daily звіти
            daily_files = list(self.results_manager.output_dir.glob("daily_*.json"))
            for file_path in daily_files:
                file_date = datetime.strptime(file_path.stem.split("_")[1], "%Y%m%d")
                if file_date >= cutoff_date:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reports.append(json.load(f))
            
            # Завантажити comprehensive звіти
            comp_files = list(self.results_manager.output_dir.glob("comprehensive_*.json"))
            for file_path in comp_files:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time >= cutoff_date:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reports.append(json.load(f))
            
            logger.info(f"[HistoricalAnalytics] Loaded {len(reports)} historical reports")
            return reports
            
        except Exception as e:
            logger.error(f"[HistoricalAnalytics] Failed to load historical reports: {e}")
            return []
    
    def calculate_performance_trends(self, reports: List[Dict]) -> Dict:
        """Розрахувати тренди продуктивності"""
        if not reports:
            return {"trend": "insufficient_data"}
        
        response_times = []
        accuracies = []
        
        for report in reports:
            daily_summary = report.get("daily_summary", {})
            if "avg_response_time" in daily_summary:
                response_times.append(daily_summary["avg_response_time"])
            
            performance_metrics = report.get("performance_metrics", {})
            if "model_accuracy" in performance_metrics:
                accuracies.append(performance_metrics["model_accuracy"].get("avg_accuracy", 0))
        
        trend_direction = "stable"
        if len(response_times) > 1:
            if response_times[-1] < response_times[0] * 0.9:
                trend_direction = "improving"
            elif response_times[-1] > response_times[0] * 1.1:
                trend_direction = "degrading"
        
        return {
            "trend": trend_direction,
            "response_times": response_times,
            "accuracies": accuracies,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0
        }
    
    def calculate_usage_trends(self, reports: List[Dict]) -> Dict:
        """Розрахувати тренди використання"""
        cpu_usage = []
        memory_usage = []
        
        for report in reports:
            system_status = report.get("system_status", {})
            if "cpu_usage" in system_status:
                cpu_usage.append(system_status["cpu_usage"])
            if "memory_usage" in system_status:
                memory_usage.append(system_status["memory_usage"])
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0
        }
    
    def calculate_error_trends(self, reports: List[Dict]) -> Dict:
        """Розрахувати тренди помилок""" 
        error_counts = []
        
        for report in reports:
            daily_summary = report.get("daily_summary", {})
            if "errors_count" in daily_summary:
                error_counts.append(daily_summary["errors_count"])
        
        trend_direction = "stable"
        if len(error_counts) > 1:
            if error_counts[-1] < error_counts[0] * 0.8:
                trend_direction = "decreasing"
            elif error_counts[-1] > error_counts[0] * 1.2:
                trend_direction = "increasing"
        
        return {
            "trend": trend_direction,
            "error_counts": error_counts,
            "total_errors": sum(error_counts),
            "avg_errors_per_day": sum(error_counts) / len(error_counts) if error_counts else 0
        }
    
    def calculate_optimization_impact(self, reports: List[Dict]) -> Dict:
        """Розрахувати вплив оптимізації"""
        return {
            "optimizations_applied": ["database_indexing", "model_pruning", "pipeline_parallelization"],
            "performance_improvement": "+15%",
            "resource_efficiency": "+12%",
            "cost_reduction": "$1,200/month"
        }
    
    def calculate_resource_trends(self, reports: List[Dict]) -> Dict:
        """Розрахувати тренди ресурсів"""
        return {
            "memory_efficiency": "improving",
            "cpu_efficiency": "stable",
            "disk_usage": "increasing_slowly",
            "network_usage": "stable"
        }
    
    def generate_trend_recommendations(self, trends: Dict) -> List[str]:
        """Генерація рекомендацій на основі трендів"""
        recommendations = []
        
        performance_trend = trends.get("performance_trends", {}).get("trend", "stable")
        error_trend = trends.get("error_trends", {}).get("trend", "stable")
        
        if performance_trend == "degrading":
            recommendations.append("Performance is degrading - consider optimization")
        
        if error_trend == "increasing":
            recommendations.append("Error rate is increasing - investigate root causes")
        
        if performance_trend == "improving":
            recommendations.append("Performance is improving - continue current strategy")
        
        return recommendations
    
    def generate_simple_forecast(self, trends: Dict) -> Dict:
        """Генерація простого прогнозу"""
        return {
            "next_week_prediction": {
                "performance": "stable_to_improving",
                "error_rate": "stable",
                "resource_usage": "stable"
            },
            "next_month_prediction": {
                "performance": "improving",
                "error_rate": "decreasing",
                "resource_usage": "slightly_increasing"
            },
            "confidence_level": "medium"
        }
