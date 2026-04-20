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
from langgraph.types import Send

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

# State — Main Graph


class AgentState(TypedDict):
    """State for the main research agent graph."""

    query: str
    plan: list  # list of SubTopic dicts
    worker_results: Annotated[list[dict], operator.add]
    report: dict  # ResearchReport as dict
    feedback: str


# Main Graph — Nodes


def plan_research(state: AgentState) -> dict:
    """LLM decomposes the user query into 2-4 parallel sub-topics."""
    planner = llm.with_structured_output(ResearchPlan)
    plan = planner.invoke(
        f"Decompose this research query into 2-4 focused sub-topics. "
        f"For each, specify a search query and which tools to use "
        f"(tavily_search for web, wikipedia for encyclopedic, arxiv for academic).\n\n"
        f"Query: {state['query']}"
    )
    return {"plan": [t.model_dump() for t in plan.sub_topics]}


def fan_out_research(state: AgentState) -> list[Send]:
    """Fan out: dispatch one worker per sub-topic using Send API."""
    return [
        Send("research_worker", {
            "topic": topic["title"],
            "search_query": topic["search_query"],
            "tools_to_use": topic["tools"],
            "findings": [],
            "summary": "",
        })
        for topic in state["plan"]
    ]


def collect_worker_result(state: WorkerState) -> dict:
    """Bridge: worker subgraph output → main graph's worker_results list."""
    return {
        "worker_results": [{
            "topic": state["topic"],
            "summary": state["summary"],
            "findings": state["findings"],
        }]
    }


# Main Graph — Build & Compile (partial — synthesis + HITL in next step)


def build_research_worker() -> StateGraph:
    """Build the research worker subgraph (tool calls + summarize)."""
    builder = StateGraph(WorkerState)
    builder.add_node("call_tools", call_tools)
    builder.add_node("summarize_findings", summarize_findings)
    builder.add_node("collect_result", collect_worker_result)
    builder.add_edge(START, "call_tools")
    builder.add_edge("call_tools", "summarize_findings")
    builder.add_edge("summarize_findings", "collect_result")
    builder.add_edge("collect_result", END)
    return builder.compile()


def build_graph():
    """Build the main research agent graph."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("plan_research", plan_research)
    builder.add_node("research_worker", build_research_worker())

    # Edges
    builder.add_edge(START, "plan_research")
    builder.add_conditional_edges("plan_research", fan_out_research)
    builder.add_edge("research_worker", END)  # temporary — synthesis added next step

    return builder.compile()


graph = build_graph()


# Smoke Test

if __name__ == "__main__":
    import json

    print("Testing full graph: plan + parallel research...\n")
    result = graph.invoke({"query": "What are the latest advances in quantum computing?"})

    print(f"Query: {result['query']}")
    print(f"Sub-topics planned: {len(result['plan'])}")
    for topic in result["plan"]:
        print(f"  • {topic['title']} → tools: {topic['tools']}")

    print(f"\nWorker results collected: {len(result['worker_results'])}")
    for wr in result["worker_results"]:
        print(f"\n--- {wr['topic']} ---")
        print(f"{wr['summary'][:300]}...")

    print("\n✓ Plan + Map-Reduce operational.")
