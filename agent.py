"""
DeepLens — Autonomous Research Agent
Built with LangGraph | Multi-tool parallel research with human-in-the-loop
"""

import os
import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, START, END

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


# State — Worker Subgraph


class WorkerState(TypedDict):
    """State for an individual research worker."""

    topic: str
    search_query: str
    tools_to_use: list[str]
    findings: Annotated[list[str], operator.add]
    summary: str


# Worker Subgraph — Nodes


def call_tools(state: WorkerState) -> dict:
    """Call each specified tool with the search query."""
    results = []
    for tool_name in state["tools_to_use"]:
        tool = tool_map.get(tool_name)
        if not tool:
            continue
        try:
            result = tool.invoke(state["search_query"])
            results.append(f"[{tool_name}]:\n{result}")
        except Exception as e:
            results.append(f"[{tool_name}]: Error — {e}")
    return {"findings": results}


def summarize_findings(state: WorkerState) -> dict:
    """LLM summarizes raw tool findings into a concise brief."""
    findings_text = "\n\n".join(state["findings"])
    response = llm.invoke(
        f"Research topic: {state['topic']}\n\n"
        f"Raw findings:\n{findings_text}\n\n"
        f"Write a concise research brief (3-5 paragraphs) with key facts and insights."
    )
    return {"summary": response.content}


# Worker Subgraph — Build & Compile

worker_builder = StateGraph(WorkerState)
worker_builder.add_node("call_tools", call_tools)
worker_builder.add_node("summarize_findings", summarize_findings)
worker_builder.add_edge(START, "call_tools")
worker_builder.add_edge("call_tools", "summarize_findings")
worker_builder.add_edge("summarize_findings", END)
worker_graph = worker_builder.compile()


# Smoke Test

if __name__ == "__main__":
    print("Testing Worker Subgraph...")
    result = worker_graph.invoke({
        "topic": "Quantum Computing Advances",
        "search_query": "quantum computing breakthroughs 2025",
        "tools_to_use": ["tavily_search", "wikipedia"],
        "findings": [],
    })
    print(f"Topic: {result['topic']}")
    print(f"Tools called: {len(result['findings'])} sources")
    print(f"\nSummary:\n{result['summary'][:500]}...")
    print("\n✓ Worker subgraph operational.")
