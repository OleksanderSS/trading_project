# src/agents/tools/pubmed_tool.py

import os
import httpx
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

async def search_clinical_trials(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    On-Demand Tool: Searches PubMed for clinical trials and medical studies.
    Requires NCBI_API_KEY in .env.
    Useful for Pharma/Biotech analysts to check drug efficacy or FDA approval odds.
    
    Args:
        query (str): The search term (e.g., "pembrolizumab clinical trial phase 3").
        max_results (int): Number of results to return.
        
    Returns:
        Dict: List of study titles and IDs.
    """
    api_key = os.getenv("NCBI_API_KEY")
    if not api_key:
        return {"error": "NCBI_API_KEY is not set in .env"}

    # Step 1: Search for IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "api_key": api_key
    }
    
    async with httpx.AsyncClient() as client:
        try:
            search_resp = await client.get(search_url, params=search_params, timeout=15.0)
            search_resp.raise_for_status()
            id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                return {"query": query, "results": "No studies found."}
                
            # Step 2: Fetch Summaries
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
                "api_key": api_key
            }
            
            sum_resp = await client.get(summary_url, params=summary_params, timeout=15.0)
            sum_resp.raise_for_status()
            summaries = sum_resp.json().get("result", {})
            
            formatted_results = []
            for uid in id_list:
                if uid in summaries:
                    doc = summaries[uid]
                    formatted_results.append({
                        "title": doc.get("title"),
                        "journal": doc.get("fulljournalname"),
                        "pubdate": doc.get("pubdate"),
                        "authors": [a.get("name") for a in doc.get("authors", [])]
                    })
                    
            return {"query": query, "studies": formatted_results}
        except Exception as e:
            return {"error": f"Failed to fetch PubMed data: {e}"}
