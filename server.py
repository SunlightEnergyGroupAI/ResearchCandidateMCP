from fastmcp import FastMCP, FastMCPApp
from ddgs import DDGS 
from apify_client import ApifyClient   
import asyncio 
from dotenv import load_dotenv  
import os  
import requests 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings 
from langchain_classic.retrievers import BM25Retriever





load_dotenv()

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



mcp.run()


             

        




