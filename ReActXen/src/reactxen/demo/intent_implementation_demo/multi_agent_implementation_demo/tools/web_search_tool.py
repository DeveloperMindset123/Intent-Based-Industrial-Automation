"""Web search tool using MCP Brave Search or langchain fallback."""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import os

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string")
    count: int = Field(default=5, description="Number of results")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the internet for information using Brave Search API"
    args_schema: type = WebSearchInput
    
    def _run(self, query: str, count: int = 5) -> str:
        """Perform web search and return formatted results."""
        try:
            from langchain_community.tools import BraveSearch
            brave_api_key = os.environ.get("BRAVE_API_KEY", "")
            if brave_api_key:
                search = BraveSearch.from_api_key(api_key=brave_api_key)
                results = search.run(query)
                return f"Search results for '{query}':\n{results[:500]}"
        except (ImportError, Exception):
            pass
        
        return f"Web search for '{query}' would return {count} results (API key not configured)"
