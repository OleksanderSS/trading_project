# src/agents/tools/comtrade_tool.py

import os
import httpx
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

load_dotenv()

def get_comtrade_key() -> str | None:
    """Attempt to load UN Comtrade key from comtrade-v1.yaml or .env"""
    # 1. Check env
    key = os.getenv("COMTRADE_API_KEY")
    if key: return key
    
    # 2. Check credentials file
    try:
        cred_path = os.path.join(os.path.dirname(__file__), "../../../credentials/comtrade-v1.yaml")
        if os.path.exists(cred_path):
            with open(cred_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("api_key")
    except Exception:
        pass
    
    # 3. Check root file
    try:
        root_path = os.path.join(os.path.dirname(__file__), "../../../comtrade-v1.yaml")
        if os.path.exists(root_path):
            with open(root_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("api_key")
    except Exception:
        pass
        
    return None

async def get_trade_volume(reporter_code: int = 842, partner_code: int = 0, cmd_code: str = "TOTAL") -> Dict[str, Any]:
    """
    On-Demand Tool: Fetches trade volumes (imports/exports) from UN Comtrade API.
    Requires API key in comtrade-v1.yaml or COMTRADE_API_KEY in .env.
    Useful for Macro Analysts checking supply chain disruptions.
    
    Args:
        reporter_code (int): Country code (e.g., 842 is USA).
        partner_code (int): Partner country code (e.g., 0 is World).
        cmd_code (str): HS commodity code (e.g., 'TOTAL').
        
    Returns:
        Dict: Trade data summary.
    """
    api_key = get_comtrade_key()
    if not api_key:
        return {"error": "Comtrade API Key not found in .env or comtrade-v1.yaml"}

    url = f"https://comtradeapi.un.org/data/v1/get/C/A/HS"
    params = {
        "reporterCode": reporter_code,
        "partnerCode": partner_code,
        "period": "2023", # Usually annual data has a lag
        "cmdCode": cmd_code,
        "flowCode": "M" # Imports
    }
    
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            
            # Comtrade returns a very nested structure, we extract the basics
            dataset = data.get("data", [])
            if not dataset:
                return {"results": "No trade data found for these parameters."}
                
            summary = []
            for item in dataset[:5]:
                summary.append({
                    "period": item.get("period"),
                    "reporter": item.get("reporterDesc"),
                    "partner": item.get("partnerDesc"),
                    "flow": item.get("flowDesc"),
                    "value_usd": item.get("primaryValue")
                })
                
            return {"source": "UN Comtrade", "data": summary}
        except Exception as e:
            return {"error": f"Failed to fetch Comtrade data: {e}"}
