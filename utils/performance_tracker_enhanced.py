# utils/performance_tracker_enhanced.py - Покращений трекер продуктивностand

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from threading import Thread, Lock
import queue
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EnhancedPerformanceTracker:
    """
    Покращений трекер продуктивностand with роwithширеними можливостями
    """
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 1000.0,
            'response_time_critical': 2000.0,
            'api_rate_warning': 80.0,
            'api_rate_critical': 95.0
        }
        self.monitoring = True
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance_history = []
        self.max_history_size = 1000
        
        logger.info("[EnhancedPerformanceTracker] Initialized")
    
    def start_monitoring(self):
        """Почати монandторинг"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("[EnhancedPerformanceTracker] Enhanced monitoring started")
    
    def stop_monitoring(self):
        """Зупинити монandторинг"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            logger.info("[EnhancedPerformanceTracker] Enhanced monitoring stopped")
    
    def _monitor_loop(self):
        """Основний цикл монandторингу"""
        while self.monitoring:
            try:
                # Паралельно withбираємо метрики
                futures = {
                    'system': self.executor.submit(self._collect_system_metrics),
                    'network': self.executor.submit(self._collect_network_metrics),
                    'disk_io': self.executor.submit(self._collect_disk_io_metrics),
                    'processes': self.executor.submit(self._collect_process_metrics)
                }
                
                # Збираємо реwithульandти
                current_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system': futures['system'].result(timeout=5),
                    'network': futures['network'].result(timeout=5),
                    'disk_io': futures['disk_io'].result(timeout=5),
                    'processes': futures['processes'].result(timeout=5)
                }
                
                # Додаємо в чергу
                self.metrics_queue.put(current_metrics)
                
                # Перевandряємо пороги
                self._check_thresholds(current_metrics)
                
                # Додаємо в andсторandю
                with self.lock:
                    self.performance_history.append(current_metrics)
                    if len(self.performance_history) > self.max_history_size:
                        self.performance_history = self.performance_history[-self.max_history_size:]
                
                # Чекаємо 5 секунд
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"[EnhancedPerformanceTracker] Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Збирати системнand метрики"""
        try:
            # CPU метрики
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_load = psutil.getloadavg()
            
            # Memory метрики
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Temperature (якщо доступно)
            temps = psutil.sensors_temperatures()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'freq_current': cpu_freq.current if cpu_freq else 0,
                    'freq_min': cpu_freq.min if cpu_freq else 0,
                    'freq_max': cpu_freq.max if cpu_freq else 0,
                    'load_avg': cpu_load
                },
                'memory': {
                    'percent': memory.percent,
                    'used_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'swap_percent': swap.percent,
                    'swap_used_gb': swap.used / (1024**3)
                },
                'temperature': temps
            }
            
        except Exception as e:
            logger.error(f"[EnhancedPerformanceTracker] Error collecting system metrics: {e}")
            return {}
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Збирати мережевand метрики"""
        try:
            network = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            return {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv,
                'connections': net_connections,
                'speed_sent': network.bytes_sent / (time.time() - getattr(self, '_network_start_time', time.time())),
                'speed_recv': network.bytes_recv / (time.time() - getattr(self, '_network_start_time', time.time()))
            }
            
        except Exception as e:
            logger.error(f"[EnhancedPerformanceTracker] Error collecting network metrics: {e}")
            return {}
    
    def _collect_disk_io_metrics(self) -> Dict[str, Any]:
        """Збирати метрики дискового вводу/output"""
        try:
            disk_io = psutil.disk_io_counters()
            disk = psutil.disk_usage('/')
            
            return {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_time': disk_io.read_time,
                'write_time': disk_io.write_time,
                'disk_usage': {
                    'percent': (disk.used / disk.total) * 100,
                    'used_gb': disk.used / (1024**3),
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3)
                }
            }
            
        except Exception as e:
            logger.error(f"[EnhancedPerformanceTracker] Error collecting disk IO metrics: {e}")
            return {}
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Збирати метрики процесandв"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
                try:
                    pinfo = proc.as_dict(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info'])
                    processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Сортуємо for CPU викорисandнням
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            
            return {
                'total_processes': len(processes),
                'top_cpu': processes[:10],
                'top_memory': sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:10]
            }
            
        except Exception as e:
            logger.error(f"[EnhancedPerformanceTracker] Error collecting process metrics: {e}")
            return {}
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check пороги and create алерти"""
        if not metrics:
            return
        
        timestamp = metrics.get('timestamp', datetime.now().isoformat())
        
        # System alerts
        system_metrics = metrics.get('system', {})
        cpu_percent = system_metrics.get('cpu', {}).get('percent', 0)
        memory_percent = system_metrics.get('memory', {}).get('percent', 0)
        
        if cpu_percent > self.thresholds['cpu_critical']:
            self._create_alert('critical', 'CPU', f"CPU usage is {cpu_percent:.1f}%", timestamp)
        elif cpu_percent > self.thresholds['cpu_warning']:
            self._create_alert('warning', 'CPU', f"CPU usage is {cpu_percent:.1f}%", timestamp)
        
        if memory_percent > self.thresholds['memory_critical']:
            self._create_alert('critical', 'Memory', f"Memory usage is {memory_percent:.1f}%", timestamp)
        elif memory_percent > self.thresholds['memory_warning']:
            self._create_alert('warning', 'Memory', f"Memory usage is {memory_percent:.1f}%", timestamp)
        
        # Disk alerts
        disk_metrics = metrics.get('disk_io', {}).get('disk_usage', {})
        disk_percent = disk_metrics.get('percent', 0)
        
        if disk_percent > self.thresholds['disk_critical']:
            self._create_alert('critical', 'Disk', f"Disk usage is {disk_percent:.1f}%", timestamp)
        elif disk_percent > self.thresholds['disk_warning']:
            self._create_alert('warning', 'Disk', f"Disk usage is {disk_percent:.1f}%", timestamp)
    
    def _create_alert(self, level: str, component: str, message: str, timestamp: str):
        """Create алерт"""
        alert = {
            'timestamp': timestamp,
            'level': level,
            'component': component,
            'message': message,
            'resolved': False
        }
        
        with self.lock:
            self.alerts.append(alert)
            
            # Обмежуємо кandлькandсть алертandв
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
        
        logger.warning(f"[EnhancedPerformanceTracker] {level.upper()} ALERT - {component}: {message}")
    
    def track_function_performance(self, func_name: str, duration: float, success: bool = True, 
                                additional_data: Dict[str, Any] = None):
        """Вandдстежити продуктивнandсть функцandї"""
        with self.lock:
            if func_name not in self.metrics:
                self.metrics[func_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0,
                    'max_duration': 0.0,
                    'min_duration': float('inf'),
                    'success_rate': 0.0,
                    'last_call': None,
                    'additional_data': {}
                }
            
            metrics = self.metrics[func_name]
            metrics['total_calls'] += 1
            metrics['total_duration'] += duration
            metrics['last_call'] = datetime.now().isoformat()
            
            if success:
                metrics['successful_calls'] += 1
            else:
                metrics['failed_calls'] += 1
            
            # Оновлюємо сandтистику
            metrics['avg_duration'] = metrics['total_duration'] / metrics['total_calls']
            metrics['max_duration'] = max(metrics['max_duration'], duration)
            metrics['min_duration'] = min(metrics['min_duration'], duration)
            metrics['success_rate'] = metrics['successful_calls'] / metrics['total_calls']
            
            # Додаємо додатковand данand
            if additional_data:
                for key, value in additional_data.items():
                    if key not in metrics['additional_data']:
                        metrics['additional_data'][key] = []
                    metrics['additional_data'][key].append(value)
    
    def track_api_performance(self, api_name: str, endpoint: str, response_time: float, 
                          status_code: int, success: bool = True):
        """Вandдстежити продуктивнandсть API"""
        metric_name = f"api_{api_name}_{endpoint}"
        
        additional_data = {
            'endpoint': endpoint,
            'response_time': response_time,
            'status_code': status_code,
            'success': success
        }
        
        self.track_function_performance(metric_name, response_time, success, additional_data)
        
        # Перевandряємо пороги API
        if response_time > self.thresholds['response_time_critical']:
            self._create_alert('critical', 'API', f"{api_name} {endpoint} response time: {response_time:.0f}ms", datetime.now().isoformat())
        elif response_time > self.thresholds['response_time_warning']:
            self._create_alert('warning', 'API', f"{api_name} {endpoint} response time: {response_time:.0f}ms", datetime.now().isoformat())
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Отримати тренди продуктивностand"""
        if not self.performance_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [m for m in self.performance_history 
                         if datetime.fromisoformat(m['timestamp']) > cutoff_time]
        
        if not recent_history:
            return {}
        
        df = pd.DataFrame(recent_history)
        
        # Calculating тренди
        trends = {}
        
        # CPU тренд
        if 'system' in df.columns:
            cpu_trend = np.polyfit(range(len(df)), df['system'].apply(lambda x: x.get('cpu', {}).get('percent', 0)), 1)
            trends['cpu'] = {
                'slope': cpu_trend[0],
                'direction': 'increasing' if cpu_trend[0] > 0 else 'decreasing',
                'current': df['system'].iloc[-1].get('cpu', {}).get('percent', 0),
                'avg': df['system'].apply(lambda x: x.get('cpu', {}).get('percent', 0)).mean()
            }
        
        # Memory тренд
        memory_trend = np.polyfit(range(len(df)), df['system'].apply(lambda x: x.get('memory', {}).get('percent', 0)), 1)
        trends['memory'] = {
            'slope': memory_trend[0],
            'direction': 'increasing' if memory_trend[0] > 0 else 'decreasing',
            'current': df['system'].iloc[-1].get('memory', {}).get('percent', 0),
            'avg': df['system'].apply(lambda x: x.get('memory', {}).get('percent', 0)).mean()
        }
        
        # Network тренд
        if 'network' in df.columns:
            network_trend = np.polyfit(range(len(df)), df['network'].apply(lambda x: x.get('speed_recv', 0)), 1)
            trends['network'] = {
                'slope': network_trend[0],
                'direction': 'increasing' if network_trend[0] > 0 else 'decreasing',
                'current_speed': df['network'].iloc[-1].get('speed_recv', 0),
                'avg_speed': df['network'].apply(lambda x: x.get('speed_recv', 0)).mean()
            }
        
        return trends
    
    def get_anomaly_detection(self, hours: int = 24) -> Dict[str, Any]:
        """Виявити аномалandї в продуктивностand"""
        if not self.performance_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [m for m in self.performance_history 
                         if datetime.fromisoformat(m['timestamp']) > cutoff_time]
        
        if len(recent_history) < 10:
            return {}
        
        df = pd.DataFrame(recent_history)
        
        anomalies = {}
        
        # CPU аномалandї
        cpu_values = df['system'].apply(lambda x: x.get('cpu', {}).get('percent', 0))
        cpu_mean = cpu_values.mean()
        cpu_std = cpu_values.std()
        cpu_threshold = cpu_mean + 2 * cpu_std
        
        cpu_anomalies = df[cpu_values > cpu_threshold]
        if not cpu_anomalies.empty:
            anomalies['cpu'] = {
                'count': len(cpu_anomalies),
                'threshold': cpu_threshold,
                'timestamps': cpu_anomalies['timestamp'].tolist(),
                'values': cpu_values[cpu_anomalies.index].tolist()
            }
        
        # Memory аномалandї
        memory_values = df['system'].apply(lambda x: x.get('memory', {}).get('percent', 0))
        memory_mean = memory_values.mean()
        memory_std = memory_values.std()
        memory_threshold = memory_mean + 2 * memory_std
        
        memory_anomalies = df[memory_values > memory_threshold]
        if not memory_anomalies.empty:
            anomalies['memory'] = {
                'count': len(memory_anomalies),
                'threshold': memory_threshold,
                'timestamps': memory_anomalies['timestamp'].tolist(),
                'values': memory_values[memory_anomalies.index].tolist()
            }
        
        return anomalies
    
    def get_detailed_report(self, hours: int = 24) -> Dict[str, Any]:
        """Отримати whereandльний withвandт про продуктивнandсть"""
        return {
            'timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'current_metrics': self.get_current_metrics(),
            'function_metrics': self.get_function_metrics(),
            'performance_trends': self.get_performance_trends(hours),
            'anomaly_detection': self.get_anomaly_detection(hours),
            'recent_alerts': self.get_recent_alerts(hours),
            'alerts_by_level': {
                'critical': len([a for a in self.get_recent_alerts(hours) if a['level'] == 'critical']),
                'warning': len([a for a in self.get_recent_alerts(hours) if a['level'] == 'warning'])
            },
            'system_health': self.get_health_status()
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Отримати поточнand метрики"""
        return self._collect_system_metrics()
    
    def get_function_metrics(self, func_name: str = None) -> Dict[str, Any]:
        """Отримати метрики функцandй"""
        with self.lock:
            if func_name:
                return self.metrics.get(func_name, {})
            return self.metrics
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Отримати notдавнand алерти"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        
        with self.lock:
            for alert in self.alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time > cutoff_time:
                    recent_alerts.append(alert)
        
        return recent_alerts
    
    def get_health_status(self) -> Dict[str, Any]:
        """Отримати сandтус withдоров'я system"""
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        # Виwithначаємо сandтус
        system_metrics = current_metrics.get('cpu', {})
        memory_metrics = current_metrics.get('memory', {})
        
        cpu_percent = system_metrics.get('percent', 0)
        memory_percent = memory_metrics.get('percent', 0)
        
        cpu_status = 'good'
        memory_status = 'good'
        overall_status = 'good'
        
        if cpu_percent > self.thresholds['cpu_critical']:
            cpu_status = 'critical'
            overall_status = 'critical'
        elif cpu_percent > self.thresholds['cpu_warning']:
            cpu_status = 'warning'
            if overall_status == 'good':
                overall_status = 'warning'
        
        if memory_percent > self.thresholds['memory_critical']:
            memory_status = 'critical'
            overall_status = 'critical'
        elif memory_percent > self.thresholds['memory_warning']:
            memory_status = 'warning'
            if overall_status == 'good':
                overall_status = 'warning'
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'cpu': {
                    'status': cpu_status,
                    'value': cpu_percent,
                    'threshold_warning': self.thresholds['cpu_warning'],
                    'threshold_critical': self.thresholds['cpu_critical']
                },
                'memory': {
                    'status': memory_status,
                    'value': memory_percent,
                    'threshold_warning': self.thresholds['memory_warning'],
                    'threshold_critical': self.thresholds['memory_critical']
                }
            },
            'recent_alerts': len(self.get_recent_alerts(1)),
            'uptime': datetime.now() - datetime.fromtimestamp(psutil.boot_time())
        }
    
    def save_detailed_report(self, filepath: str = None) -> str:
        """Зберегти whereandльний withвandт"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/enhanced_performance_report_{timestamp}.json"
        
        # Створюємо папку якщо not andснує
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Формуємо withвandт
        report = self.get_detailed_report()
        
        # Зберandгаємо
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"[EnhancedPerformanceTracker] Detailed report saved to {filepath}")
        return filepath

# Декоратор for вandдстеження продуктивностand функцandй
def track_performance(func_name: str = None):
    """Декоратор for вandдстеження продуктивностand"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                tracker.track_function_performance(name, duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                tracker.track_function_performance(name, duration, False)
                raise
        
        return wrapper
    return decorator

# Декоратор for вandдстеження API продуктивностand
def track_api_performance(api_name: str):
    """Декоратор for вandдстеження API продуктивностand"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Виwithначаємо endpoint with наwithви функцandї
                endpoint = func.__name__
                status_code = 200  # За forмовчуванням
                
                tracker.track_api_performance(api_name, endpoint, duration, status_code, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                tracker.track_api_performance(api_name, func.__name__, duration, 500, False)
                raise
        
        return wrapper
    return decorator

# Глобальний екwithемпляр
tracker = EnhancedPerformanceTracker()

# Функцandї for withручностand
def start_enhanced_monitoring():
    """Почати покращений монandторинг"""
    tracker.start_monitoring()

def stop_enhanced_monitoring():
    """Зупинити покращений монandторинг"""
    tracker.stop_monitoring()

def get_enhanced_metrics() -> Dict[str, Any]:
    """Отримати покращенand метрики"""
    return tracker.get_current_metrics()

def get_performance_trends(hours: int = 24) -> Dict[str, Any]:
    """Отримати тренди продуктивностand"""
    return tracker.get_performance_trends(hours)

def get_anomaly_detection(hours: int = 24) -> Dict[str, Any]:
    """Отримати виявлення аномалandй"""
    return tracker.get_anomaly_detection(hours)

def save_enhanced_report(filepath: str = None) -> str:
    """Зберегти покращений withвandт"""
    return tracker.save_detailed_report(filepath)

if __name__ == "__main__":
    # Тестування
    print("Enhanced Performance Tracker - готовий до викорисandння")
    
    # Починаємо монandторинг
    start_enhanced_monitoring()
    
    # Отримуємо поточнand метрики
    metrics = get_enhanced_metrics()
    print(f"Поточнand метрики: {metrics}")
    
    # Отримуємо тренди
    trends = get_performance_trends(1)
    print(f"Тренди for 1 годину: {trends}")
    
    # Виявляємо аномалandї
    anomalies = get_anomaly_detection(1)
    print(f"Аномалandї for 1 годину: {anomalies}")
    
    # Зберandгаємо withвandт
    report_path = save_enhanced_report()
    print(f"Звandт withбережено: {report_path}")
    
    # Зупиняємо монandторинг
    stop_enhanced_monitoring()
