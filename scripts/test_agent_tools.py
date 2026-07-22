import asyncio
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.tools.weather_tool import check_weather
from src.agents.tools.gdelt_tool import search_global_events
from src.agents.tools.pubmed_tool import search_clinical_trials
from src.agents.tools.eia_tool import get_oil_prices
from src.agents.tools.comtrade_tool import get_trade_volume

async def main():
    print("🚀 Тестування інструментів (Agent Tools)...")
    
    print("\n🌤️ 1. Open-Meteo (Погода в Техасі):")
    weather = await check_weather(29.76, -95.36, days_forecast=1)
    print(weather)
    
    print("\n📰 2. GDELT (Події: 'sanctions'):")
    gdelt = await search_global_events("sanctions", max_records=2)
    print(gdelt)
    
    print("\n🏥 3. PubMed (Медицина: 'pembrolizumab'):")
    pubmed = await search_clinical_trials("pembrolizumab", max_results=2)
    print(pubmed)
    
    print("\n🛢️ 4. EIA (Ціни на нафту):")
    eia = await get_oil_prices(days_back=2)
    print(eia)
    
    print("\n🚢 5. UN Comtrade (Торгівля: Зерно з України - 804):")
    comtrade = await get_trade_volume(reporter_code=804)
    print(comtrade)

if __name__ == "__main__":
    asyncio.run(main())
