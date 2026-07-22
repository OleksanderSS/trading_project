"""
Тестовий скрипт для перевірки альтернативних даних колекторів
Перевіряємо чи вони працюють і чи дають реальні дані
"""
import asyncio
import sys
from pathlib import Path

# Додаємо проект в path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.data.collectors.vix_collector import VIXCollector
from src.data.collectors.put_call_ratio_collector import PutCallRatioCollector
from src.data.collectors.fear_greed_collector import FearGreedCollector
from src.data.collectors.aaii_sentiment_collector import AIISentimentCollector
from src.data.collectors.cftc_collector import CFTCCollector
from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector
from src.data.collectors.insider_collector import InsiderCollector


async def test_vix_collector():
    """Тестуємо VIX Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО VIX COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'vix_data',
        'hash_keys': ['date', 'vix_current', 'volatility_regime'],
        'params': {
            'period': '30d',
            'interval': '1d'
        },
        'ticker': '^VIX'
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = VIXCollector(configs, http_factory, db_manager)
    
    try:
        data = await collector.collect_data()
        
        if data:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, VIX: {record.get('vix_close')}, Регім: {record.get('volatility_regime')}")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_put_call_ratio_collector():
    """Тестуємо Put/Call Ratio Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО PUT/CALL RATIO COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'put_call_ratio_data',
        'hash_keys': ['date', 'put_call_ratio', 'sentiment_signal'],
        'allow_sample_fallback': False  # НЕ дозволяємо synthetic fallback
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = PutCallRatioCollector(configs, http_factory, db_manager)
    
    try:
        data = await collector.collect_data()
        
        if data:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Ratio: {record.get('put_call_ratio')}, Sentiment: {record.get('sentiment_classification')}")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
                synthetic_count = sum(1 for r in data if r.get('is_synthetic'))
                print(f"   Синтетичних записів: {synthetic_count}/{len(data)}")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_fear_greed_collector():
    """Тестуємо Fear & Greed Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО FEAR & GREED COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'fear_greed_data',
        'hash_keys': ['date', 'fear_greed_index', 'classification']
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = FearGreedCollector(configs, http_factory, db_manager)
    
    try:
        data = await collector.collect_data()
        
        if data:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Value: {record.get('value')}, Category: {record.get('fear_greed_category')}")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_aaii_collector():
    """Тестуємо AAII Sentiment Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО AAII SENTIMENT COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'aaii_sentiment_data',
        'hash_keys': ['date', 'bullish', 'bearish', 'neutral']
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = AIISentimentCollector(configs, http_factory, db_manager)
    
    try:
        data = await collector.collect_data()
        
        if data:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Bullish: {record.get('bullish')}%, Bearish: {record.get('bearish')}%")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_cftc_collector():
    """Тестуємо CFTC Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО CFTC COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'cftc_data',
        'hash_keys': ['date', 'instrument', 'net_position'],
        'allow_sample_fallback': False  # НЕ дозволяємо synthetic fallback
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = CFTCCollector(configs, http_factory, db_manager)
    
    try:
        data = await collector.collect_data()
        
        if data:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Інструмент: {record.get('instrument')}, Net Position: {record.get('net_position')}")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
                synthetic_count = sum(1 for r in data if r.get('is_synthetic'))
                print(f"   Синтетичних записів: {synthetic_count}/{len(data)}")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_economic_calendar_collector():
    """Тестуємо Economic Calendar Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО ECONOMIC CALENDAR COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'economic_calendar',
        'api_url': 'https://ec.forexprostools.com/api.php',
        'headers': {
            'User-Agent': 'Mozilla/5.0'
        },
        'request_payload': {
            'country': [25, 32, 37, 143, 72],  # US, UK, EU, Japan, China
            'importance': [1, 2, 3],
            'timeZone': 25
        },
        'api_mappings': {
            'country': {
                'US': 25,
                'UK': 32,
                'EU': 37,
                'Japan': 143,
                'China': 72
            },
            'impact': {
                'low': 1,
                'medium': 2,
                'high': 3
            }
        },
        'column_names': ['time', 'currency', 'impact', 'event', 'actual', 'forecast', 'previous'],
        'countries': ['US', 'UK', 'EU', 'Japan', 'China'],
        'importance': ['low', 'medium', 'high'],
        'days_back': 7,
        'days_ahead': 30
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = EconomicCalendarCollector(configs, http_factory, db_manager)
    
    try:
        data = await collector.collect_data()
        
        if data:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Timestamp: {record.get('timestamp')}, Event: {record.get('event')}, Impact: {record.get('impact')}")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_insider_collector():
    """Тестуємо Insider Collector"""
    print("\n" + "="*60)
    print("🔍 ТЕСТУЄМО INSIDER COLLECTOR")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'insider_trades',
        'hash_keys': ['filing_date', 'ticker', 'insider_name'],
        'urls': ['http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=730&fdlt1=&fdgt1=&fdr=730&td=0&tdr=0&sd=199&sdr=0&sdlt=0&sdgt=0&sy=-o&sos=&sot=&col=trd&sh=0&sw=2'],
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'column_mapping': {
            'col_0': 'filing_date',
            'col_1': 'ticker',
            'col_2': 'insider_name',
            'col_3': 'title',
            'col_4': 'trade_type',
            'col_5': 'price',
            'col_6': 'quantity',
            'col_7': 'owned',
            'col_8': 'delta_own',
            'col_9': 'value'
        }
    }
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    http_factory = HttpClientFactory(config_manager, error_handler)
    db_manager = DataManager(config_manager, error_handler)
    
    collector = InsiderCollector(configs, http_factory, db_manager)
    
    try:
        # InsiderCollector має два методи run() - використовуємо той, що повертає list[dict]
        data = await collector.run(tickers=None)
        
        if data and len(data) > 0:
            print(f"✅ Успішно отримано {len(data)} записів")
            print(f"📊 Приклад даних:")
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('filing_date')}, Ticker: {record.get('ticker')}, Insider: {record.get('insider_name')}")
            
            # Перевіряємо на синтетичність
            has_synthetic = any('is_synthetic' in r for r in data)
            if has_synthetic:
                print("⚠️  Виявлено синтетичні дані!")
            else:
                print("✅ Дані виглядають реальними")
                
            return True, data
        else:
            print("❌ Не отримано даних")
            return False, None
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def main():
    """Головна функція для тестування всіх колекторів"""
    print("\n" + "="*60)
    print("🧪 ТЕСТУВАННЯ АЛЬТЕРНАТИВНИХ ДАНИХ КОЛЕКТОРІВ")
    print("="*60)
    
    results = {}
    
    # Тестуємо кожен колектор
    results['VIX'] = await test_vix_collector()
    results['Put/Call Ratio'] = await test_put_call_ratio_collector()
    results['Fear & Greed'] = await test_fear_greed_collector()
    results['AAII'] = await test_aaii_collector()
    results['CFTC'] = await test_cftc_collector()
    results['Economic Calendar'] = await test_economic_calendar_collector()
    results['Insider'] = await test_insider_collector()
    
    # Підсумок
    print("\n" + "="*60)
    print("📊 ПІДСУМКИ ТЕСТУВАННЯ")
    print("="*60)
    
    for collector_name, (success, data) in results.items():
        status = "✅ ПРАЦЮЄ" if success else "❌ НЕ ПРАЦЮЄ"
        data_count = f"({len(data)} записів)" if data else ""
        print(f"{collector_name:20s}: {status} {data_count}")
    
    working = sum(1 for s, _ in results.values() if s)
    total = len(results)
    print(f"\nВсього працює: {working}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
