from fastmcp import FastMCP, FastMCPApp
from ddgs import DDGS 
from apify_client import ApifyClient   
import asyncio 
from dotenv import load_dotenv  
import os  
import requests

load_dotenv()

mcp = FastMCP()   

# bing_description = """ 
# Searches bing search engine, and provides the information found from the following sources

# Args: 
# search_query: str (the query given to the search engine)
# """ 

# yandex_description = """ 
# Searches yandex search engine, and provides the information found from the following sources

# Args: 
# search_query: str (the query given to the search engine)
# """ 


linkedinResearch_description = """ 
Searches LinkedIn, and returns candidate information 

Args: 
first_name: str, 
last_name: str, 
schools: list[str] | None = []


""" 

# comboSearch_description = """ 
# Searches yandex and bing search engine, and provides the information found from the following sources

# Args: 
# search_query: str (the query given to both search engines)


# """





# @mcp.tool(description=bing_description) 
# async def bing_search(search_query: str) -> list[str]:
#     information = [] 
#     async with AsyncWebCrawler() as crawler: 
#         bing_results = DDGS().text(search_query, backend="bing") 
#         for result in bing_results: 
#             url = result.get("href") 
#             result = await crawler.arun(url) 
#             information.append(result.markdown) 
#     return information 

# @mcp.tool(description=yandex_description) 
# async def yandex_search(search_query: str) -> list[str]: 
#     information = [] 
#     async with AsyncWebCrawler() as crawler: 
#             bing_results = DDGS().text(search_query, backend="yandex") 
#             for result in bing_results: 
#                 url = result.get("href") 
#                 result = await crawler.arun(url) 
#                 information.append(result.markdown) 
#     return information  


# @mcp.tool(description=comboSearch_description) 
# async def comboSearch(search_query: str) -> list[str]: 
#     return await asyncio.run(yandex_search(search_query)) + await asyncio.run(bing_search(search_query))


@mcp.tool(description=linkedinResearch_description) 
async def linkedInSearch(first_name: str, last_name: str, schools: list[str] | None = []) -> dict: 
    client = ApifyClient(os.getenv("APIFY_API_KEY")) 
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



mcp.run(transport='http')


             

        




