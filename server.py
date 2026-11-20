# Jims Candidate MCP server
# v2.3
from fastmcp import FastMCP
from apify_client import ApifyClient, ApifyClientAsync
from dotenv import load_dotenv
from ddgs import DDGS
from tavily import AsyncTavilyClient
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
from functools import cache
from datetime import date
import numpy as np
import httpx
import trafilatura
import os
import re
import sys
import asyncio 
import requests


load_dotenv()
apifyKey = os.getenv("APIFY_API_KEY")
tavilyKey = os.getenv("TAVILY_API_KEY")
hunterKey = os.getenv("HUNTERIO_API_KEY")

mcp = FastMCP("Candidate Research")

searchEngines = ["yahoo", "yandex", "bing", "wikipedia"]
resultsPerEngine = 10
pagesToFetch = 8
maxChunksPerPage = 15
maxResultsPerPage = 2
rrfK = 60
rerankCandidates = 30
fetchTimeout = 10
fetchHeaders = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

#Cost: $0.10 per search page + $0.004 per full profile, so 25 profiles is about $0.20 a search
#Filters sent to the actor (titles, locations, headcount, searchQuery) are free, profiles scraped are not
#An energy_search runs the actor 3 times, so it costs about $0.60
profileSearchActor = "M2FMdjRVeF1HPGFcc"
profilesToScan = 25
candidatesToReturn = 5
minGapMonths = 3
experienceBuckets = {"1": (0, 1), "2": (1, 2), "3": (3, 5), "4": (6, 10), "5": (10, 60)} #actor yearsOfExperienceIds
smallMidHeadcounts = ["B", "C", "D", "E", "F"] #1 to 1,000 employees
territoryStates = ["Maryland", "Delaware", "New Jersey", "Pennsylvania", "New York", "Connecticut", "Massachusetts", "Rhode Island"]
energySearchQuery = '"energy procurement" OR "energy broker" OR "retail energy" OR "electricity supply" OR "natural gas supply" OR "deregulated"'
energyTitles = [
    "Energy Consultant", "Energy Advisor", "Energy Broker", "Energy Account Executive",
    "Energy Account Manager", "Energy Sales Representative", "Energy Procurement Specialist",
    "Energy Procurement Manager", "Energy Sales Consultant",
]
energyKeywords = [
    "energy procurement", "energy broker", "energy brokerage", "retail energy", "electricity supply",
    "natural gas supply", "electric and gas", "power and gas", "deregulated", "commercial energy",
    "energy consultant", "energy advisor", "constellation", "nrg", "direct energy", "engie",
    "freepoint", "smartestenergy", "smartest energy", "calpine", "agera", "spark energy",
]
monthNumbers = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
monthNames = list(monthNumbers.keys())

textSplitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

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
Searches Tavily, Yahoo, Yandex, Bing and Wikipedia, reads the top pages, and returns the most relevant passages ranked with hybrid search (BM25 + embeddings) and a cross encoder reranker

Args:
- search_query: str (what to search for e.g "Jane Doe Sunlight Energy Group")
- top_k: int (how many passages to return, default 5)

Returns:
- list of Documents with page_content (page title + passage) and metadata (url, engine, title, rerank_score)
"""

findSalesCandidates_description = """
Recruiting tool: finds salespeople on LinkedIn in SEG's territory (MD, DE, NJ, PA, NY, CT, MA, RI) by current job title and years of experience. Leaves out anyone outside the requested locations or experience range, with large employment gaps, or who changes jobs too often, and checks each person's full work history for energy procurement. Returns up to 5 candidates, energy matches first.

Args:
- job_titles: list[str] (current job titles to search e.g ["Energy Consultant", "Account Executive"])
- locations: list[str] | None (states or cities inside SEG's territory e.g ["Maryland", "Baltimore"], default is the whole territory)
- min_years: int (minimum years of experience, default 0)
- max_years: int (maximum years of experience, default 40)
- energy_search: bool (also searches people currently in energy sales titles like "Energy Consultant", people in the job titles who held an energy title before, and the job titles with energy procurement terms on their profile; use for Account Executive and Account Manager, default False)
- require_energy: bool (only return people whose work history matched energy procurement, default False)
- small_mid_companies_only: bool (only people whose current company has 1,000 employees or fewer, default True)
- max_gap_months: int (leave out anyone with an employment gap longer than this many months, default 6)
- min_average_tenure_months: int (leave out anyone whose average time per company is shorter than this many months, default 12)

Returns:
- list of up to 5 dicts with name, headline, current_role, location, linkedin_url, open_to_work, years_experience, companies, average_tenure_months, longest_gap_months, gaps (each gap has from, to and months) and energy_evidence (each matched job as "title at company (matched 'keyword')", flagged when the job mentions solar)
"""

findEmail_description = """
Recruiting tool: finds a candidate's work email with Hunter.io from their name and company

Args:
- first_name: str
- last_name: str
- company_domain: str | None (company website domain e.g "sunlightenergygroup.com", most accurate)
- company: str | None (company name when the domain is unknown e.g "Sunlight Energy Group")

Returns:
- dict with email, score (confidence 0-100), position, company, domain, verification and sources, or an error
""" 

# instantSearchDescription = """  
# # Search engine that provides instant results however the search query must be basic as it is good for Entity summaries 

# # Args: 
# # - search_query: str 

# # Returns: 
# # - list[str] (information returned)
# # """
# @mcp.tool(description=instantSearchDescription, tags={"research"})
# def useInstantSearch(search_query: str) -> tuple[str]:   
#     url = f"https://api.duckduckgo.com/?q={search_query}&format=json"
#     response = requests.get(url) 
#     data = response.json() 
#     abstract_info = data.get("Abstract", "NOTHING FOUND") 
#     source = data.get("AbstractSource", "NOTHING FOUND") 
#     return (abstract_info, source)
#     ...

@mcp.tool(description=linkedinResearch_description, tags={"research"})
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


async def searchTavily(query: str) -> list[Document]:
    try:
        tavilyClient = AsyncTavilyClient(api_key=tavilyKey)
        response = await tavilyClient.search(query, max_results=resultsPerEngine, include_raw_content="text")
    except Exception as e:
        print(f"tavily search failed: {e}", file=sys.stderr)
        return []
    documents = []
    for result in response["results"]:
        pageContent = f"{result['title']}\n{result['content']}"
        metadata = {"url": result["url"], "engine": "tavily", "page_text": result.get("raw_content") or ""}
        documents.append(Document(page_content=pageContent, metadata=metadata))
    return documents


def searchEngine(query: str, engine: str) -> list[Document]:
    try:
        engineResults = DDGS().text(query, backend=engine, max_results=resultsPerEngine)
    except Exception as e:
        print(f"{engine} search failed: {e}", file=sys.stderr)
        return []
    documents = []
    for result in engineResults:
        pageContent = f"{result.get('title', '')}\n{result.get('body', '')}"
        metadata = {"url": result.get("href", ""), "engine": engine, "page_text": ""}
        documents.append(Document(page_content=pageContent, metadata=metadata))
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


async def searchAllEngines(query: str) -> list[Document]:
    # tavily goes first so its copy of a duplicate url, which has the page text, is the one kept
    engineSearches = [searchTavily(query)]
    for engine in searchEngines:
        engineSearches.append(asyncio.to_thread(searchEngine, query, engine))
    engineResults = await asyncio.gather(*engineSearches)
    documents = []
    for results in engineResults:
        documents.extend(results)
    return removeDuplicates(documents)


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


async def fetchPageText(httpClient: httpx.AsyncClient, url: str) -> str:
    try:
        response = await httpClient.get(url)
        if response.status_code != 200 or "html" not in response.headers.get("content-type", ""):
            return ""
        pageText = await asyncio.to_thread(trafilatura.extract, response.text)
        return pageText or ""
    except Exception as e:
        print(f"fetch failed for {url}: {e}", file=sys.stderr)
        return ""


async def fetchMissingPages(documents: list[Document]) -> list[Document]:
    missingPages = [document for document in documents if not document.metadata["page_text"]]
    async with httpx.AsyncClient(timeout=fetchTimeout, follow_redirects=True, headers=fetchHeaders) as httpClient:
        pageTexts = await asyncio.gather(*[fetchPageText(httpClient, document.metadata["url"]) for document in missingPages])
    for document, pageText in zip(missingPages, pageTexts):
        document.metadata["page_text"] = pageText
    return documents


def chunkPages(documents: list[Document]) -> list[Document]:
    """
    Splits each page into passages with the page title on top so the rankers know
    where a passage came from. Pages that could not be read (blocked, PDF, timeout)
    fall back to their search snippet so they still get ranked.
    """
    chunks = []
    for document in documents:
        title = document.page_content.split("\n")[0]
        metadata = {"url": document.metadata["url"], "engine": document.metadata["engine"], "title": title}
        pageText = document.metadata["page_text"]
        if not pageText:
            chunks.append(Document(page_content=document.page_content, metadata=metadata))
            continue
        splittedText = textSplitter.split_text(pageText)[:maxChunksPerPage]
        for chunk in splittedText:
            chunks.append(Document(page_content=f"{title}\n{chunk}", metadata=metadata.copy()))
    return chunks


def rerankDocuments(query: str, documents: list[Document]) -> list[Document]:
    texts = [document.page_content for document in documents]
    rerankScores = list(getRerankerModel().rerank(query, texts))
    for document, score in zip(documents, rerankScores):
        document.metadata["rerank_score"] = round(float(score), 4)
    return sorted(documents, key=lambda document: document.metadata["rerank_score"], reverse=True)


def pickTopResults(documents: list[Document], top_k: int) -> list[Document]:
    # caps passages per page so one long page can't fill every slot
    pageCounts = Counter()
    topResults = []
    for document in documents:
        url = document.metadata["url"]
        if pageCounts[url] >= maxResultsPerPage:
            continue
        pageCounts[url] += 1
        topResults.append(document)
        if len(topResults) == top_k:
            break
    return topResults


@mcp.tool(description=megaSearch_description, tags={"research"})
async def megaSearch(search_query: str, top_k: int = 5) -> list[Document]:
    try:
        documents = await searchAllEngines(search_query)
        if not documents:
            return []
        rankedSnippets = await asyncio.to_thread(hybridSearch, search_query, documents)
        topPages = await fetchMissingPages(rankedSnippets[:pagesToFetch])
        chunks = chunkPages(topPages)
        rankedChunks = await asyncio.to_thread(hybridSearch, search_query, chunks)
        rerankedChunks = await asyncio.to_thread(rerankDocuments, search_query, rankedChunks[:rerankCandidates])
        return pickTopResults(rerankedChunks, top_k)
    except Exception as e:
        return [Document(page_content=f"Error occurred: {e}", metadata={})] 




@mcp.tool(description=findEmail_description, tags={"recruiting"})
async def findEmail(first_name: str, last_name: str, company_domain: str | None = None, company: str | None = None) -> dict:
    if not company_domain and not company:
        return {"error": "Provide company_domain or company"}
    params = {"first_name": first_name, "last_name": last_name, "api_key": hunterKey}
    if company_domain:
        params["domain"] = company_domain
    else:
        params["company"] = company
    try:
        async with httpx.AsyncClient(timeout=20) as httpClient:
            response = await httpClient.get("https://api.hunter.io/v2/email-finder", params=params)
        responseJson = response.json()
        if response.status_code != 200:
            return {"error": responseJson["errors"][0]["details"]}
        emailData = responseJson["data"]
        if not emailData["email"]:
            return {"error": f"No email found for {first_name} {last_name} at {emailData['domain'] or company}"}
        sources = [source["uri"] for source in emailData["sources"]]
        verification = emailData["verification"] or {}
        return {
            "email": emailData["email"],
            "score": emailData["score"],
            "position": emailData["position"],
            "company": emailData["company"],
            "domain": emailData["domain"],
            "verification": verification.get("status"),
            "sources": sources
        }
    except Exception as e:
        return {"error": f"Error occurred: {e}"}


def toMonthIndex(jobDate: dict | None, isEnd: bool) -> int | None:
    if not jobDate:
        return None
    if jobDate.get("text") == "Present":
        today = date.today()
        return today.year * 12 + today.month
    if not jobDate.get("year"):
        return None
    # year only dates count as the whole year
    defaultMonth = 12 if isEnd else 1
    month = monthNumbers.get(jobDate.get("month"), defaultMonth)
    return jobDate["year"] * 12 + month


def monthIndexToText(monthIndex: int) -> str:
    year = (monthIndex - 1) // 12
    month = (monthIndex - 1) % 12
    return f"{monthNames[month]} {year}"


def getJobPeriod(job: dict) -> list[int] | None:
    start = toMonthIndex(job.get("startDate"), isEnd=False)
    end = toMonthIndex(job.get("endDate"), isEnd=True)
    if start and end and end >= start:
        return [start, end]
    return None


def mergePeriods(periods: list[list[int]]) -> list[list[int]]:
    mergedPeriods = []
    for period in sorted(periods):
        if mergedPeriods and period[0] <= mergedPeriods[-1][1] + 1:
            mergedPeriods[-1][1] = max(mergedPeriods[-1][1], period[1])
        else:
            mergedPeriods.append(list(period))
    return mergedPeriods


def getEmploymentPeriods(profile: dict) -> list[list[int]]:
    """
    Turns the experience section into month ranges and merges overlapping jobs,
    so two jobs held at the same time don't hide or fake a gap.
    """
    periods = []
    for job in profile.get("experience") or []:
        period = getJobPeriod(job)
        if period:
            periods.append(period)
    return mergePeriods(periods)


def getCompanyTenures(profile: dict) -> list[int]:
    """
    Months spent at each company. Roles at the same company are combined so a
    promotion isn't counted as a job change, and internships are skipped since
    they are short by design.
    """
    companyPeriods = {}
    for job in profile.get("experience") or []:
        if "intern" in (job.get("employmentType") or "").lower():
            continue
        period = getJobPeriod(job)
        if period:
            companyPeriods.setdefault(job.get("companyName"), []).append(period)
    tenures = []
    for periods in companyPeriods.values():
        tenures.append(sum(end - start + 1 for start, end in mergePeriods(periods)))
    return tenures


def findGaps(periods: list[list[int]]) -> list[dict]:
    today = date.today()
    currentMonth = today.year * 12 + today.month
    gaps = []
    for previousPeriod, nextPeriod in zip(periods, periods[1:]):
        gapMonths = nextPeriod[0] - previousPeriod[1] - 1
        if gapMonths >= minGapMonths:
            gaps.append({"from": monthIndexToText(previousPeriod[1]), "to": monthIndexToText(nextPeriod[0]), "months": gapMonths})
    if periods and currentMonth - periods[-1][1] - 1 >= minGapMonths:
        gaps.append({"from": monthIndexToText(periods[-1][1]), "to": "Present", "months": currentMonth - periods[-1][1] - 1})
    return gaps


def getCurrentRole(profile: dict) -> str:
    for job in profile.get("experience") or []:
        endDate = job.get("endDate") or {}
        if endDate.get("text") == "Present":
            return f"{job.get('position')} at {job.get('companyName')}"
    return "No current role listed"


def findEnergyEvidence(profile: dict) -> list[str]:
    """
    Checks every job in the work history, not just the current one, for energy
    procurement terms and supplier names. Solar sales reps are often titled
    "Energy Consultant", so any match that mentions solar is flagged for Eloka.
    """
    energyEvidence = []
    for job in profile.get("experience") or []:
        jobText = f"{job.get('position')} {job.get('companyName')} {job.get('description')}".lower()
        for keyword in energyKeywords:
            if keyword not in jobText:
                continue
            evidence = f"{job.get('position')} at {job.get('companyName')} (matched '{keyword}')"
            if "solar" in jobText:
                evidence += " - mentions solar"
            energyEvidence.append(evidence)
            break
    return energyEvidence


def summarizeCandidate(profile: dict) -> dict:
    periods = getEmploymentPeriods(profile)
    gaps = findGaps(periods)
    monthsWorked = sum(end - start + 1 for start, end in periods)
    longestGap = max([gap["months"] for gap in gaps], default=0)
    tenures = getCompanyTenures(profile)
    averageTenure = round(sum(tenures) / len(tenures)) if tenures else 0
    location = profile.get("location") or {}
    parsedLocation = location.get("parsed") or {}
    return {
        "name": f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
        "headline": profile.get("headline"),
        "current_role": getCurrentRole(profile),
        "location": location.get("linkedinText"),
        "linkedin_url": profile.get("linkedinUrl"),
        "open_to_work": profile.get("openToWork"),
        "country_code": location.get("countryCode"),
        "state": parsedLocation.get("state"),
        "years_experience": round(monthsWorked / 12, 1),
        "companies": len(tenures),
        "average_tenure_months": averageTenure,
        "longest_gap_months": longestGap,
        "gaps": gaps,
        "energy_evidence": findEnergyEvidence(profile)
    }


def isInTerritory(candidate: dict) -> bool:
    if candidate["country_code"] != "US":
        return False
    # metro areas like "Washington DC-Baltimore Area" come back with no state, so they are kept
    if candidate["state"] and candidate["state"] not in territoryStates:
        return False
    return True


def isInRequestedLocations(candidate: dict, locations: list[str] | None) -> bool:
    # cities and metro areas can't be checked against a state, so only state searches are narrowed
    requestedStates = [location for location in locations or [] if location in territoryStates]
    if not requestedStates or not candidate["state"]:
        return True
    return candidate["state"] in requestedStates


def isGoodCandidate(candidate: dict, locations: list[str] | None, max_gap_months: int, min_average_tenure_months: int, min_years: int, max_years: int, require_energy: bool) -> bool:
    if not isInTerritory(candidate):
        return False
    if not isInRequestedLocations(candidate, locations):
        return False
    if candidate["years_experience"] < min_years or candidate["years_experience"] > max_years:
        return False
    if candidate["longest_gap_months"] > max_gap_months:
        return False
    if candidate["average_tenure_months"] < min_average_tenure_months:
        return False
    if require_energy and not candidate["energy_evidence"]:
        return False
    return True


def pickExperienceIds(min_years: int, max_years: int) -> list[str]:
    experienceIds = []
    for bucketId, (low, high) in experienceBuckets.items():
        if low <= max_years and high >= min_years:
            experienceIds.append(bucketId)
    return experienceIds


def buildSearchInputs(job_titles: list[str], locations: list[str] | None, min_years: int, max_years: int, energy_search: bool, small_mid_companies_only: bool) -> list[dict]:
    """
    One actor run normally. An energy_search does three runs, since energy reps are
    rarely titled "Account Executive": people currently in energy sales titles, people
    in the job titles who held an energy title before, and the job titles with energy
    procurement terms anywhere on the profile.
    """
    baseInput = {
        "profileScraperMode": "Full",
        "locations": locations or territoryStates,
        "yearsOfExperienceIds": pickExperienceIds(min_years, max_years),
        "maxItems": profilesToScan
    }
    if small_mid_companies_only:
        baseInput["companyHeadcount"] = smallMidHeadcounts
    if not energy_search:
        return [{**baseInput, "currentJobTitles": job_titles}]
    return [
        {**baseInput, "currentJobTitles": energyTitles},
        {**baseInput, "currentJobTitles": job_titles, "pastJobTitles": energyTitles},
        {**baseInput, "currentJobTitles": job_titles, "searchQuery": energySearchQuery}
    ]


async def runProfileSearch(client: ApifyClientAsync, run_input: dict) -> list[dict]:
    run = await client.actor(profileSearchActor).call(run_input=run_input)
    profiles = []
    async for profile in client.dataset(run["defaultDatasetId"]).iterate_items():
        profiles.append(profile)
    return profiles


@mcp.tool(description=findSalesCandidates_description, tags={"recruiting"})
async def findSalesCandidates(job_titles: list[str], locations: list[str] | None = None, min_years: int = 0, max_years: int = 40, energy_search: bool = False, require_energy: bool = False, small_mid_companies_only: bool = True, max_gap_months: int = 6, min_average_tenure_months: int = 12) -> list[dict]:
    if min_years > max_years:
        return [{"error": f"min_years ({min_years}) is bigger than max_years ({max_years})"}]
    try:
        client = ApifyClientAsync(token=apifyKey)
        searchInputs = buildSearchInputs(job_titles, locations, min_years, max_years, energy_search, small_mid_companies_only)
        searchResults = await asyncio.gather(*[runProfileSearch(client, run_input) for run_input in searchInputs])
        candidates = []
        seenUrls = set()
        for profiles in searchResults:
            for profile in profiles:
                candidate = summarizeCandidate(profile)
                if candidate["linkedin_url"] in seenUrls:
                    continue
                seenUrls.add(candidate["linkedin_url"])
                if not isGoodCandidate(candidate, locations, max_gap_months, min_average_tenure_months, min_years, max_years, require_energy):
                    continue
                candidates.append(candidate)
        candidates.sort(key=lambda candidate: len(candidate["energy_evidence"]), reverse=True)
        return candidates[:candidatesToReturn]
    except Exception as e:
        return [{"error": f"Error occurred: {e}"}]


if __name__ == "__main__":
    print(asyncio.run(megaSearch("Sunlight Energy Group solar")))