# Jims Candidate MCP server
from fastmcp import FastMCP
from apify_client import ApifyClient
from dotenv import load_dotenv
from ddgs import DDGS
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document
from functools import cache
import numpy as np
import os
import re
import sys
import asyncio


load_dotenv()
apifyKey = os.getenv("APIFY_API_KEY")

mcp = FastMCP("Candidate Research")

searchEngines = ["yahoo", "yandex", "bing", "wikipedia"]
resultsPerEngine = 10
rrfK = 60
rerankCandidates = 20

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

megaSearch_description = """
Searches Yahoo, Yandex, Bing and Wikipedia, then ranks the combined results with hybrid search (BM25 + embeddings) and a cross encoder reranker

Args:
- search_query: str (what to search for e.g "Jane Doe Sunlight Energy Group")
- top_k: int (how many results to return, default 5)

Returns:
- list of Documents with page_content (title + snippet) and metadata (url, engine, rerank_score)
"""


@mcp.tool(description=linkedinResearch_description)
async def linkedInSearch(first_name: str, last_name: str, schools: list[str] | None = [], locations: list[str] | None = []) -> list[dict]:
    client = ApifyClient(token=apifyKey)
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


# loaded on first megaSearch call so the server starts fast
@cache
def getEmbeddingsModel() -> TextEmbedding:
    return TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


@cache
def getRerankerModel() -> TextCrossEncoder:
    return TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")


def embedTexts(texts: list[str]) -> np.ndarray:
    embeddings = np.array(list(getEmbeddingsModel().embed(texts)))
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def searchEngine(query: str, engine: str) -> list[Document]:
    try:
        engineResults = DDGS().text(query, backend=engine, max_results=resultsPerEngine)
    except Exception as e:
        print(f"{engine} search failed: {e}", file=sys.stderr)
        return []
    documents = []
    for result in engineResults:
        pageContent = f"{result.get('title', '')}\n{result.get('body', '')}"
        documents.append(Document(page_content=pageContent, metadata={"url": result.get("href", ""), "engine": engine}))
    return documents


def removeDuplicates(documents: list[Document]) -> list[Document]:
    seenUrls = set()
    uniqueDocuments = []
    for document in documents:
        url = document.metadata["url"]
        if url in seenUrls:
            continue
        seenUrls.add(url)
        uniqueDocuments.append(document)
    return uniqueDocuments


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rankPositions(scores: list[float]) -> list[int]:
    sortedIndexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positions = [0] * len(scores)
    for rank, index in enumerate(sortedIndexes):
        positions[index] = rank
    return positions


def hybridSearch(query: str, documents: list[Document]) -> list[Document]:
    """
    Scores every document two ways, BM25 for exact keyword matches (names, companies)
    and embedding similarity for meaning, then merges the two rankings with
    reciprocal rank fusion so neither score scale dominates the other.
    """
    texts = [document.page_content for document in documents]

    bm25 = BM25Okapi([tokenize(text) for text in texts])
    bm25Scores = bm25.get_scores(tokenize(query)).tolist()

    queryEmbedding = embedTexts([query])[0]
    documentEmbeddings = embedTexts(texts)
    denseScores = (documentEmbeddings @ queryEmbedding).tolist()

    bm25Ranks = rankPositions(bm25Scores)
    denseRanks = rankPositions(denseScores)
    fusedScores = []
    for bm25Rank, denseRank in zip(bm25Ranks, denseRanks):
        fusedScores.append(1 / (rrfK + bm25Rank + 1) + 1 / (rrfK + denseRank + 1))

    rankedIndexes = sorted(range(len(documents)), key=lambda index: fusedScores[index], reverse=True)
    return [documents[index] for index in rankedIndexes]


def rerankDocuments(query: str, documents: list[Document], top_k: int) -> list[Document]:
    texts = [document.page_content for document in documents]
    rerankScores = list(getRerankerModel().rerank(query, texts))
    for document, score in zip(documents, rerankScores):
        document.metadata["rerank_score"] = round(float(score), 4)
    rerankedDocuments = sorted(documents, key=lambda document: document.metadata["rerank_score"], reverse=True)
    return rerankedDocuments[:top_k]


@mcp.tool(description=megaSearch_description)
async def megaSearch(search_query: str, top_k: int = 5) -> list[Document]:
    try:
        engineSearches = [asyncio.to_thread(searchEngine, search_query, engine) for engine in searchEngines]
        engineResults = await asyncio.gather(*engineSearches)
        documents = []
        for results in engineResults:
            documents.extend(results)
        documents = removeDuplicates(documents)
        if not documents:
            return []
        rankedDocuments = await asyncio.to_thread(hybridSearch, search_query, documents)
        candidates = rankedDocuments[:rerankCandidates]
        return await asyncio.to_thread(rerankDocuments, search_query, candidates, top_k)
    except Exception as e:
        return [Document(page_content=f"Error occurred: {e}", metadata={})]


if __name__ == "__main__":
    print(asyncio.run(megaSearch("Sunlight Energy Group solar")))
