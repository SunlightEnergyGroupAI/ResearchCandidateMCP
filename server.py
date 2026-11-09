from fastmcp import FastMCP, FastMCPApp
from apify_client import ApifyClient   
import os  
import requests 
import asyncio



#co
APIKEY = "apify_api_g1sUfOUOUoz66mERxN1r9LCRUTObgl35hOxa"

mcp = FastMCP("Candidate Research")    

linkedinResearch_description = """ 
Searches LinkedIn, and returns candidate information 

Args: 
first_name: str, 
last_name: str, 
schools: list[str] | None = [] 
locations: list[str] | None = []   

Return: 
dict


""" 

@mcp.tool(description=linkedinResearch_description) 
async def linkedInSearch(first_name: str, last_name: str, schools: list[str] | None = [], locations: list[str] | None = []) -> list[dict]: 
    client = ApifyClient(token="apify_api_g1sUfOUOUoz66mERxN1r9LCRUTObgl35hOxa") 
    candidate_info = []
    run_input = {
        "profileScraperMode": "Full",
        "firstName": first_name,
        "lastName": last_name,
        "strictSearch": True,
        "maxPages": 1, 
        "maxItems": 5, 
        "schools": schools,
        "locations": locations
    }
    run = client.actor("CP1SVZfEwWflrmWCX").call(run_input=run_input)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items(): 
        candidate_info.append(item)
    return candidate_info





             

        




