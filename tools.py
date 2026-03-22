import os
from tavily import TavilyClient

def search(query):
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query)
    # Format the results for better readability
    formatted_results = ""
    if "results" in results:
        for result in results["results"]:
            formatted_results += f"Title: {result.get('title', 'N/A')}\n"
            formatted_results += f"URL: {result.get('url', 'N/A')}\n"
            formatted_results += f"Content: {result.get('content', 'N/A')}\n\n"
    return formatted_results