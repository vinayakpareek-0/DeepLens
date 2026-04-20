"""
DeepLens — Autonomous Research Agent
Built with LangGraph | Multi-tool parallel research with human-in-the-loop
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# LLM Configuration

llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0,
)

# Tools

tavily_search = TavilySearchResults(max_results=3, name="tavily_search")

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000),
    name="wikipedia",
)

arxiv = ArxivQueryRun(name="arxiv")

tools = [tavily_search, wikipedia, arxiv]
tool_map = {tool.name: tool for tool in tools}

# Pydantic Schemas — Input & Planning

class SubTopic(BaseModel):
    """A single research sub-topic identified by the planner."""

    title: str = Field(description="Short title for this sub-topic")
    search_query: str = Field(description="Optimized search query")
    tools: list[Literal["tavily_search", "wikipedia", "arxiv"]] = Field(
        description="Tools to use for this sub-topic"
    )


class ResearchPlan(BaseModel):
    """LLM's decomposition of a query into parallel sub-topics."""

    sub_topics: list[SubTopic] = Field(
        description="2-4 sub-topics to research in parallel",
        min_length=2,
        max_length=4,
    )

# Pydantic Schemas — Output & Report

class Source(BaseModel):
    """A source citation."""

    title: str = Field(description="Source title or page name")
    url: str = Field(default="", description="URL if available")


class ReportSection(BaseModel):
    """One section of the final report."""

    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content in markdown")


class ResearchReport(BaseModel):
    """The final structured research report."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary (2-3 sentences)")
    sections: list[ReportSection] = Field(description="Report sections")
    sources: list[Source] = Field(description="All sources cited")


# Smoke Test

if __name__ == "__main__":
    # Test LLM connection
    print("Testing LLM...")
    response = llm.invoke("Say 'DeepLens is online' in one sentence.")
    print(f"LLM: {response.content}\n")

    # Test structured output
    print("Testing structured output (ResearchPlan)...")
    planner = llm.with_structured_output(ResearchPlan)
    plan = planner.invoke("What are the latest advances in quantum computing?")
    print(f"Plan: {plan.model_dump_json(indent=2)}\n")

    # Test tools
    print("Testing Tavily...")
    print(tavily_search.invoke("latest quantum computing breakthroughs 2025")[:200])
    print("\nTesting Wikipedia...")
    print(wikipedia.invoke("Quantum computing")[:200])
    print("\nTesting ArXiv...")
    print(arxiv.invoke("quantum computing")[:200])

    print("\n✓ All systems operational.")
