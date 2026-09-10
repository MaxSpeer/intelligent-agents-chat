"""Search uploaded documents and return cited passages within a server-bound project."""

from __future__ import annotations

from intelligent_agents_chat import documents
from intelligent_agents_chat.tools import Tool

DEFAULT_TOP_K = 5
MAX_TOP_K = 10

_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search this project's uploaded documents (not the web) for passages "
            "relevant to a query, and return them with citations (document name, "
            "location). Use this whenever a question might be answered by a "
            "document the user uploaded to this project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for -- a question or a few keywords.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"How many passages to return (default {DEFAULT_TOP_K}).",
                },
            },
            "required": ["query"],
        },
    },
}


def _format_results(results) -> str:
    """Format passages as Markdown with a title, locator, and text for each result."""
    blocks = [f"### {result.title}\n{result.locator}\n\n{result.text}" for result in results]
    return "\n\n---\n\n".join(blocks)


def build_tool(project_id: str) -> Tool:
    """Build a document-search tool scoped to `project_id`."""

    async def run(arguments: dict) -> str:
        query = (arguments.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        top_k = arguments.get("top_k")
        try:
            limit = min(MAX_TOP_K, max(1, int(top_k))) if top_k is not None else DEFAULT_TOP_K
        except (TypeError, ValueError):
            limit = DEFAULT_TOP_K

        try:
            results = await documents.rag_retriever.retrieve(
                project_id=project_id, text=query, limit=limit
            )
        except Exception as error:
            return f"Error: document search failed: {error}"

        if not results:
            return "No matching passages found in this project's documents."
        return _format_results(results)

    return Tool(name=_SCHEMA["function"]["name"], schema=_SCHEMA, run=run)
