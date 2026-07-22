# src/agents/tools/gdelt_tool.py

import httpx
from typing import Dict, Any

async def search_global_events(query: str, max_records: int = 10) -> Dict[str, Any]:
    """
    On-Demand Tool: Searches the GDELT 2.0 API for recent global events (protests, sanctions, etc).
    Useful for geopolitical analysts verifying news rumors.
    
    Args:
        query (str): The search term (e.g., "protest Iran" or "sanctions Russia").
        max_records (int): Maximum number of articles to return.
        
    Returns:
        Dict: A list of recent events matching the query.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json"
    }
    
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                response = await client.get(url, params=params, timeout=15.0)
                if response.status_code == 429:
                    import asyncio
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
                
                # Format nicely for the LLM
                formatted_results = []
                for art in articles:
                    formatted_results.append({
                        "title": art.get("title"),
                        "url": art.get("url"),
                        "domain": art.get("domain"),
                        "date": art.get("seendate")
                    })
                return {"query": query, "events": formatted_results}
            except Exception as e:
                if attempt == 2:
                    return {"error": f"Failed to fetch GDELT events: {e}"}
        return {"error": "GDELT Rate Limit Exceeded"}
