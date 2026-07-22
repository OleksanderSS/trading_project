"""
Тестуємо кожен колектор окремо для детального аналізу помилок
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.data.collectors.put_call_ratio_collector import PutCallRatioCollector
from src.data.collectors.fear_greed_collector import FearGreedCollector
from src.data.collectors.aaii_sentiment_collector import AIISentimentCollector
from src.data.collectors.cftc_collector import CFTCCollector
from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector


async def test_put_call():
    """Тестуємо Put/Call Ratio"""
    print("\n" + "="*60)
    print("🔍 PUT/CALL RATIO")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'put_call_ratio_data',
        'hash_keys': ['date', 'put_call_ratio', 'sentiment_signal'],
        'allow_sample_fallback': False
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
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Ratio: {record.get('put_call_ratio')}")
        else:
            print("❌ Не отримано даних")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


async def test_fear_greed():
    """Тестуємо Fear & Greed"""
    print("\n" + "="*60)
    print("🔍 FEAR & GREED")
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
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Value: {record.get('value')}")
        else:
            print("❌ Не отримано даних")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


async def test_aaii():
    """Тестуємо AAII"""
    print("\n" + "="*60)
    print("🔍 AAII SENTIMENT")
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
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Bullish: {record.get('bullish')}%")
        else:
            print("❌ Не отримано даних")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


async def test_cftc():
    """Тестуємо CFTC"""
    print("\n" + "="*60)
    print("🔍 CFTC")
    print("="*60)
    
    configs = {
        'enabled': True,
        'timeout': 30,
        'table_name': 'cftc_data',
        'hash_keys': ['date', 'instrument', 'net_position'],
        'allow_sample_fallback': False
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
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Дата: {record.get('date')}, Інструмент: {record.get('instrument')}")
        else:
            print("❌ Не отримано даних")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


async def test_economic_calendar():
    """Тестуємо Economic Calendar"""
    print("\n" + "="*60)
    print("🔍 ECONOMIC CALENDAR")
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
            'country': [25, 32, 37, 143, 72],
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
            for i, record in enumerate(data[:3]):
                print(f"   {i+1}. Timestamp: {record.get('timestamp')}, Event: {record.get('event')}")
        else:
            print("❌ Не отримано даних")
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Тестуємо кожен колектор окремо"""
    await test_put_call()
    await test_fear_greed()
    await test_aaii()
    await test_cftc()
    await test_economic_calendar()


if __name__ == "__main__":
    asyncio.run(main())
