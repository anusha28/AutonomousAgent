"""
Custom MCP tools for the research agent.

Tools:
  • web_search    — searches for research papers, PDFs, and articles via Tavily
  • download_pdfs — downloads PDF URLs to ./Research Papers/ and extracts text
"""

import os
import re
import json
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

load_dotenv()

# ── PDF URL resolver ──────────────────────────────────────────────────────────

def _resolve_pdf_url(url: str) -> str | None:
    """
    Convert a known academic article page URL to its direct PDF download URL.
    Returns the PDF URL if the pattern is recognised, otherwise None.

    Supported sources:
      • PMC     https://pmc.ncbi.nlm.nih.gov/articles/PMC{id}/
      • PubMed  https://pubmed.ncbi.nlm.nih.gov/{id}/   (redirects to PMC PDF)
      • arXiv   https://arxiv.org/abs/{id}
      • bioRxiv https://www.biorxiv.org/content/{path}
      • medRxiv https://www.medrxiv.org/content/{path}
      • eLife   https://elifesciences.org/articles/{id}
    """
    # Already a direct PDF
    if url.lower().endswith(".pdf"):
        return url

    # PMC: /articles/PMCxxxxxxx[/] → /articles/PMCxxxxxxx/pdf/
    m = re.search(r"(pmc\.ncbi\.nlm\.nih\.gov/articles/PMC\d+)", url)
    if m:
        base = "https://" + m.group(1).rstrip("/")
        return base + "/pdf/"

    # PubMed abstract → resolve to PMC PDF via known redirect pattern
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if m:
        pmid = m.group(1)
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"   # kept as-is; download will follow redirect

    # arXiv abstract → PDF
    m = re.search(r"arxiv\.org/abs/([^\s/?]+)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"

    # bioRxiv / medRxiv article page → full PDF
    m = re.search(r"(biorxiv\.org|medrxiv\.org)/content/([^\s?#]+?)(?:v\d+)?$", url)
    if m:
        return f"https://www.{m.group(1)}/content/{m.group(2)}.full.pdf"

    # eLife
    m = re.search(r"elifesciences\.org/articles/(\d+)", url)
    if m:
        return f"https://elifesciences.org/articles/{m.group(1)}.pdf"

    return None


# ── Tavily client ─────────────────────────────────────────────────────────────

def _tavily_client() -> AsyncTavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. "
            "Copy .env.example to .env and add your key from https://app.tavily.com"
        )
    return AsyncTavilyClient(api_key=api_key)


# ── Tool 1: web_search ────────────────────────────────────────────────────────

@tool(
    "web_search",
    (
        "Search the web for recent, credible research papers, PDFs, and articles "
        "on a given topic. Returns titles, URLs, content snippets, and relevance "
        "scores. Each result includes a resolved pdf_url where possible (PMC, arXiv, "
        "bioRxiv, eLife, etc. are auto-converted to their direct PDF download URLs). "
        "Pass the pdf_urls list to the download_pdfs tool to save them locally."
    ),
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research topic or question to search for.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-20). Default 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    query: str = args["query"]
    max_results: int = args.get("max_results", 10)

    client = _tavily_client()

    response = await client.search(
        query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_domains=[
            "pmc.ncbi.nlm.nih.gov",
            "pubmed.ncbi.nlm.nih.gov",
            "arxiv.org",
            "biorxiv.org",
            "medrxiv.org",
            "elifesciences.org",
            "nature.com",
            "science.org",
            "cell.com",
            "journals.physiology.org",
            "jnnutrition.org",
            "ajcn.nutrition.org",
            "nutritionandmetabolism.biomedcentral.com",
            "f1000research.com",
        ],
    )

    results = response.get("results", [])

    formatted: list[dict] = []
    pdf_urls: list[str] = []

    for r in results:
        url: str = r.get("url", "")
        pdf_url = _resolve_pdf_url(url)
        if pdf_url and pdf_url not in pdf_urls:
            pdf_urls.append(pdf_url)

        formatted.append({
            "title": r.get("title", ""),
            "url": url,
            "pdf_url": pdf_url,
            "score": round(r.get("score", 0.0), 4),
            "snippet": r.get("content", "")[:400],
        })

    output = {
        "query": query,
        "total_results": len(formatted),
        "pdf_urls": pdf_urls,   # resolved PDF download URLs — pass these to download_pdfs
        "results": formatted,
    }

    return {
        "content": [{"type": "text", "text": json.dumps(output, indent=2)}]
    }


# ── Tool 2: download_pdfs ─────────────────────────────────────────────────────

RESEARCH_PAPERS_DIR = Path(__file__).parent / "Research Papers"

def _safe_filename(url: str) -> str:
    """Derive a safe filename from a URL."""
    # PMC pdf/ URLs: extract the PMC ID from the path
    m = re.search(r"PMC(\d+)", url)
    if m:
        return f"PMC{m.group(1)}.pdf"

    # arXiv: last path segment is the paper ID
    m = re.search(r"arxiv\.org/pdf/([^\s/?]+?)(?:\.pdf)?$", url)
    if m:
        return f"arxiv_{re.sub(r'[^\w]', '_', m.group(1))}.pdf"

    # Generic fallback
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    name = re.sub(r"[^\w\-.]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name or "paper.pdf"


@tool(
    "download_pdfs",
    (
        "Downloads PDFs from the given URLs into the local 'Research Papers' folder "
        "and extracts their text content. Pass the pdf_urls list from web_search results. "
        "Returns the filename, page count, and a text excerpt for each successfully "
        "downloaded PDF."
    ),
    {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of PDF URLs to download.",
            },
        },
        "required": ["urls"],
    },
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True),
)
async def download_pdfs(args: dict[str, Any]) -> dict[str, Any]:
    urls: list[str] = args["urls"]
    RESEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as http:
        for url in urls:
            filename = _safe_filename(url)
            dest = RESEARCH_PAPERS_DIR / filename
            entry: dict[str, Any] = {"url": url, "filename": filename}

            try:
                response = await http.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type:
                    entry["status"] = "skipped"
                    entry["reason"] = f"Not a PDF — content-type: {content_type}"
                    results.append(entry)
                    continue

                dest.write_bytes(response.content)
                entry["saved_to"] = str(dest)

                # Extract text with pypdf
                reader = PdfReader(dest)
                page_count = len(reader.pages)
                text_parts = []
                for page in reader.pages[:5]:  # preview first 5 pages
                    text_parts.append(page.extract_text() or "")
                preview_text = "\n".join(text_parts).strip()

                entry["status"] = "success"
                entry["pages"] = page_count
                entry["text_preview"] = preview_text[:1000]

            except httpx.HTTPStatusError as e:
                entry["status"] = "error"
                entry["reason"] = f"HTTP {e.response.status_code}"
            except Exception as e:
                entry["status"] = "error"
                entry["reason"] = str(e)

            results.append(entry)

    summary = {
        "downloaded": sum(1 for r in results if r.get("status") == "success"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "saved_to_folder": str(RESEARCH_PAPERS_DIR),
        "files": results,
    }

    return {
        "content": [{"type": "text", "text": json.dumps(summary, indent=2)}]
    }


# ── MCP server ────────────────────────────────────────────────────────────────

research_tools_server = create_sdk_mcp_server(
    name="research_tools",
    version="1.0.0",
    tools=[web_search, download_pdfs],
)
