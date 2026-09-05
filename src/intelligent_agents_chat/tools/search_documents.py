"""A tool that searches the current project's uploaded documents (see
context.py's document_service/rag_retriever, and app.py's "Manage project
documents" dialog for how they get there) and returns the most relevant
passages, with citations.

Unlike every other tool in this package, this one needs to know which
project the calling conversation belongs to -- and that has to come from the
server, not the model (a project_id the model could set itself would let it
read another project's documents just by asking). So this module doesn't
export a static module-level TOOL like calculator/subagent/webfetch/
websearch do; it exports build_tool(project_id), a factory chat.py's
stream_reply calls once per turn with the real project_id already bound into
a closure, before the model ever sees the tool.

This also means retrieval only ever needs a single tool call, not a whole
separate automatic pipeline: the model decides for itself whether a question
needs a document search, and can call this more than once (e.g. to refine
the query) within the normal tool-call loop -- see chat.py's stream_reply.
"""

from __future__ import annotations

from intelligent_agents_chat.retrieval import RetrievalQuery
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
    blocks = []
    for result in results:
        blocks.append(f"[{result.title}] ({result.locator})\n{result.text}")
    return "\n\n".join(blocks)


def build_tool(project_id: str) -> Tool:
    """A search_documents Tool bound to one project -- see the module
    docstring for why this is a factory instead of a static TOOL.
    """

    async def run(arguments: dict) -> str:
        query = (arguments.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        top_k = arguments.get("top_k")
        try:
            limit = min(MAX_TOP_K, max(1, int(top_k))) if top_k is not None else DEFAULT_TOP_K
        except (TypeError, ValueError):
            limit = DEFAULT_TOP_K

        # Imported here, not at module level: context.py builds rag_retriever
        # at import time from the real repository/embedding gateway, and this
        # avoids importing that whole bootstrap just to read one name at
        # import time, before it may be ready.
        from intelligent_agents_chat.context import rag_retriever

        try:
            results = await rag_retriever.retrieve(
                RetrievalQuery(project_id=project_id, text=query, limit=limit)
            )
        except Exception as error:
            return f"Error: document search failed: {error}"

        if not results:
            return "No matching passages found in this project's documents."
        return _format_results(results)

    return Tool(name=_SCHEMA["function"]["name"], schema=_SCHEMA, run=run)
