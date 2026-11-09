from fastmcp import FastMCP, FastMCPApp
from apify_client import ApifyClient   
import os  
import requests 




APIFY_API_KEY = "apify_api_g1sUfOUOUoz66mERxN1r9LCRUTObgl35hOxa"


mcp = FastMCP("Candidate Research")    

linkedinResearch_description = """ 
Searches LinkedIn, and returns candidate information 

Args: 
first_name: str, 
last_name: str, 
schools: list[str] | None = []


""" 

@mcp.tool(description=linkedinResearch_description) 
async def linkedInSearch(first_name: str, last_name: str, schools: list[str] | None = []) -> dict: 
    client = ApifyClient(APIFY_API_KEY) 
    candidate_info = []
    run_input = {
        "profileScraperMode": "Full",
        "firstName": first_name,
        "lastName": last_name,
        "strictSearch": True,
        "maxPages": 1, 
        "schools": schools,
        "maxItems": 1
    }
    run = client.actor("CP1SVZfEwWflrmWCX").call(run_input=run_input)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items(): 
        candidate_info.append(item)
    return candidate_info[0] 



mcp.run()


             

        




