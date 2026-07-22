# src/agents/tools/weather_tool.py

import httpx
from typing import Dict, Any

async def check_weather(latitude: float, longitude: float, days_forecast: int = 3) -> Dict[str, Any]:
    """
    On-Demand Tool: Fetches weather forecast for a specific location using Open-Meteo.
    Useful for assessing physical risks (hurricanes, freezes) to energy or agricultural assets.
    
    Args:
        latitude (float): Latitude of the location (e.g., 29.76 for Houston).
        longitude (float): Longitude of the location (e.g., -95.36 for Houston).
        days_forecast (int): Number of days to forecast (default 3).
        
    Returns:
        Dict: Weather data including temperature, wind speed, and precipitation.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "auto",
        "forecast_days": days_forecast
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return {
                "location": {"lat": latitude, "lon": longitude},
                "forecast": data.get("daily", {})
            }
        except Exception as e:
            return {"error": f"Failed to fetch weather: {e}"}
