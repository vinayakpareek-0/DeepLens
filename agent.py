"""
DeepLens — Autonomous Research Agent
Built with LangGraph | Multi-tool parallel research with human-in-the-loop
"""

import os
import json
import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt

load_dotenv()

# LLM
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


# Pydantic Schemas
class SubTopic(BaseModel):
    """A single research sub-topic."""
    title: str = Field(description="Short title for this sub-topic")
    search_query: str = Field(description="Optimized search query")
    tools: list[Literal["tavily_search", "wikipedia", "arxiv"]] = Field(
        description="Tools to use for this sub-topic"
    )

class ResearchPlan(BaseModel):
    """LLM's decomposition of a query into parallel sub-topics."""
    sub_topics: list[SubTopic] = Field(
        description="2-4 sub-topics to research in parallel",
        min_length=2, max_length=4,
    )

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


# State Schemas
class WorkerState(TypedDict):
    """State for an individual research worker."""
    topic: str
    search_query: str
    tools_to_use: list[str]
    findings: Annotated[list[str], operator.add]
    summary: str

class AgentState(TypedDict):
    """State for the main research agent graph."""
    query: str
    plan: list
    worker_results: Annotated[list[dict], operator.add]
    report: dict
    feedback: str


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
            results.append(f"[{tool_name}]: Error - {e}")
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

def collect_worker_result(state: WorkerState) -> dict:
    """Bridge: worker output -> main graph's worker_results list."""
    return {"worker_results": [{
        "topic": state["topic"],
        "summary": state["summary"],
        "findings": state["findings"],
    }]}


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
    """Fan out: dispatch one worker per sub-topic via Send API."""
    return [
        Send("research_worker", {
            "topic": t["title"],
            "search_query": t["search_query"],
            "tools_to_use": t["tools"],
            "findings": [],
            "summary": "",
        })
        for t in state["plan"]
    ]

def synthesize_report(state: AgentState) -> dict:
    """Reduce: merge all worker results into a structured ResearchReport."""
    results_text = "\n\n".join(
        f"## {wr['topic']}\n{wr['summary']}" for wr in state["worker_results"]
    )
    synthesizer = llm.with_structured_output(ResearchReport)
    report = synthesizer.invoke(
        f"You are a research report writer. Synthesize these research findings "
        f"into a well-structured report with a title, executive summary, "
        f"detailed sections, and source citations.\n\n"
        f"Original query: {state['query']}\n\n"
        f"Research findings:\n{results_text}"
    )
    return {"report": report.model_dump()}

def human_review(state: AgentState) -> dict:
    """Pause for human review. User can approve or request edits."""
    report = state["report"]
    print("\n" + "=" * 60)
    print("REPORT READY FOR REVIEW")
    print("=" * 60)
    print(f"\nTitle: {report['title']}")
    print(f"Summary: {report['summary']}")
    print(f"Sections: {len(report['sections'])}")
    print(f"Sources: {len(report['sources'])}")
    print("\n" + "=" * 60)

    feedback = interrupt(
        "Review the report above. Reply 'approve' to save, or provide edit instructions."
    )

    if feedback.lower().strip() == "approve":
        return {"feedback": "approved"}

    # Re-synthesize with feedback
    synthesizer = llm.with_structured_output(ResearchReport)
    revised = synthesizer.invoke(
        f"Revise this research report based on feedback.\n\n"
        f"Current report:\n{json.dumps(report, indent=2)}\n\n"
        f"Feedback: {feedback}"
    )
    return {"report": revised.model_dump(), "feedback": feedback}

def should_continue_review(state: AgentState) -> str:
    """Route: if approved go to save, otherwise loop back for review."""
    if state.get("feedback") == "approved":
        return "save_report"
    return "human_review"

def save_report(state: AgentState) -> dict:
    """Save the final approved report."""
    print("\nReport approved and saved.")
    return {}


# Graph Assembly
def build_research_worker():
    """Build the research worker subgraph."""
    builder = StateGraph(WorkerState)
    builder.add_node("call_tools", call_tools)
    builder.add_node("summarize_findings", summarize_findings)
    builder.add_node("collect_result", collect_worker_result)
    builder.add_edge(START, "call_tools")
    builder.add_edge("call_tools", "summarize_findings")
    builder.add_edge("summarize_findings", "collect_result")
    builder.add_edge("collect_result", END)
    return builder.compile()

def build_graph(checkpointer=None):
    """Build the main research agent graph."""
    builder = StateGraph(AgentState)
    builder.add_node("plan_research", plan_research)
    builder.add_node("research_worker", build_research_worker())
    builder.add_node("synthesize_report", synthesize_report)
    builder.add_node("human_review", human_review)
    builder.add_node("save_report", save_report)

    builder.add_edge(START, "plan_research")
    builder.add_conditional_edges("plan_research", fan_out_research)
    builder.add_edge("research_worker", "synthesize_report")
    builder.add_edge("synthesize_report", "human_review")
    builder.add_conditional_edges("human_review", should_continue_review)
    builder.add_edge("save_report", END)

    return builder.compile(checkpointer=checkpointer)

graph = build_graph()


# Smoke Test
if __name__ == "__main__":
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    memory = InMemorySaver()

    test_graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-1"}}

    print("Running DeepLens...\n")

    # First run — pauses at human_review interrupt
    for event in test_graph.stream(
        {"query": "What are the latest advances in quantum computing?"},
        config=config,
    ):
        for node in event:
            if node != "__interrupt__":
                print(f"[{node}] done")

    # Show report preview
    state = test_graph.get_state(config)
    report = state.values.get("report", {})
    print(f"\nReport: {report.get('title', 'N/A')}")
    print(f"Sections: {len(report.get('sections', []))}")
    print(f"Sources: {len(report.get('sources', []))}")

    # Resume with approval
    print("\nApproving report...")
    for event in test_graph.stream(Command(resume="approve"), config=config):
        for node in event:
            if node != "__interrupt__":
                print(f"[{node}] done")

    print("\nDone.")

