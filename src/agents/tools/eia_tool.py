# src/agents/tools/eia_tool.py

import os
import httpx
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

async def get_oil_prices(days_back: int = 5) -> Dict[str, Any]:
    """
    On-Demand Tool: Fetches recent spot prices for Crude Oil (WTI) from the EIA API.
    Requires EIA_API_KEY in .env.
    Useful for Energy Analysts forecasting commodity inflation.
    
    Args:
        days_back (int): Number of recent data points to fetch.
        
    Returns:
        Dict: List of recent dates and prices.
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        return {"error": "EIA_API_KEY is not set in .env"}

    # EIA API v2 endpoint for spot prices
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "frequency": "daily",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": days_back,
        "api_key": api_key
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("response", {}).get("data", [])
            formatted = []
            for rec in records:
                formatted.append({
                    "date": rec.get("period"),
                    "product": rec.get("product-name"),
                    "price": rec.get("value"),
                    "units": rec.get("units")
                })
                
            return {"source": "EIA", "data": formatted}
        except Exception as e:
            return {"error": f"Failed to fetch EIA data: {e}"}
