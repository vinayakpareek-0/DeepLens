"""
DeepLens — Autonomous Research Agent
Built with LangGraph | Multi-tool parallel research with human-in-the-loop
"""

import os
import json
import uuid
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
from langgraph.store.memory import InMemoryStore

load_dotenv()


# --- LLM & Tools ---

llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0,
)

tavily_search = TavilySearchResults(max_results=3, name="tavily_search")
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000),
    name="wikipedia",
)
arxiv = ArxivQueryRun(name="arxiv")

tools = [tavily_search, wikipedia, arxiv]
tool_map = {tool.name: tool for tool in tools}


# --- Pydantic Schemas ---

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


# --- State Schemas ---

class WorkerState(TypedDict):
    """State for an individual research worker (subgraph)."""
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


# --- Worker Subgraph Nodes ---

def call_tools(state: WorkerState) -> dict:
    """Call each tool specified for this sub-topic."""
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
    """LLM summarizes raw tool output into a research brief."""
    findings_text = "\n\n".join(state["findings"])
    response = llm.invoke(
        f"Research topic: {state['topic']}\n\n"
        f"Raw findings:\n{findings_text}\n\n"
        f"Write a concise research brief (3-5 paragraphs) with key facts and insights."
    )
    return {"summary": response.content}

def collect_worker_result(state: WorkerState) -> dict:
    """Bridge: pack worker output into parent graph's worker_results list."""
    return {"worker_results": [{
        "topic": state["topic"],
        "summary": state["summary"],
        "findings": state["findings"],
    }]}


# --- Main Graph Nodes ---

def plan_research(state: AgentState) -> dict:
    """LLM decomposes user query into 2-4 parallel sub-topics."""
    planner = llm.with_structured_output(ResearchPlan)
    plan = planner.invoke(
        f"Decompose this research query into 2-4 focused sub-topics. "
        f"For each, specify a search query and which tools to use "
        f"(tavily_search for web, wikipedia for encyclopedic, arxiv for academic).\n\n"
        f"Query: {state['query']}"
    )
    return {"plan": [t.model_dump() for t in plan.sub_topics]}

def fan_out_research(state: AgentState) -> list[Send]:
    """Map: dispatch one worker per sub-topic via Send API."""
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
        f"Synthesize these research findings into a structured report "
        f"with title, executive summary, detailed sections, and sources.\n\n"
        f"Original query: {state['query']}\n\n"
        f"Research findings:\n{results_text}"
    )
    return {"report": report.model_dump()}

def human_review(state: AgentState) -> dict:
    """Pause execution for human approval or edit instructions."""
    report = state["report"]
    feedback = interrupt({
        "type": "review",
        "title": report["title"],
        "summary": report["summary"],
        "sections": len(report["sections"]),
        "sources": len(report["sources"]),
        "message": "Reply 'approve' to save, or provide edit instructions.",
    })
    if feedback.lower().strip() in ("approve", "yes", "y"):
        return {"feedback": "approved"}
    # Include original research data for better revision
    results_context = "\n".join(
        f"- {wr['topic']}: {wr['summary'][:200]}" for wr in state.get("worker_results", [])
    )
    synthesizer = llm.with_structured_output(ResearchReport)
    revised = synthesizer.invoke(
        f"Revise this research report based on user feedback. "
        f"Use the original research data to add more detail.\n\n"
        f"Current report:\n{json.dumps(report, indent=2)}\n\n"
        f"Original research data:\n{results_context}\n\n"
        f"User feedback: {feedback}"
    )
    return {"report": revised.model_dump(), "feedback": feedback}

def should_continue_review(state: AgentState) -> str:
    """Route: approved -> save, otherwise -> review again."""
    return "save_report" if state.get("feedback") == "approved" else "human_review"

def save_report(state: AgentState, *, store) -> dict:
    """Save approved report to long-term memory store."""
    store.put(
        ("research",),
        state["query"][:50],
        {"query": state["query"], "report": state["report"]},
    )
    return {}


# --- Graph Assembly ---

def build_worker_subgraph():
    """Build the research worker subgraph: call_tools -> summarize -> collect."""
    builder = StateGraph(WorkerState)
    builder.add_node("call_tools", call_tools)
    builder.add_node("summarize_findings", summarize_findings)
    builder.add_node("collect_result", collect_worker_result)
    builder.add_edge(START, "call_tools")
    builder.add_edge("call_tools", "summarize_findings")
    builder.add_edge("summarize_findings", "collect_result")
    builder.add_edge("collect_result", END)
    return builder.compile()

def build_graph(checkpointer=None, store=None):
    """Build the main research agent graph."""
    builder = StateGraph(AgentState)
    builder.add_node("plan_research", plan_research)
    builder.add_node("research_worker", build_worker_subgraph())
    builder.add_node("synthesize_report", synthesize_report)
    builder.add_node("human_review", human_review)
    builder.add_node("save_report", save_report)
    builder.add_edge(START, "plan_research")
    builder.add_conditional_edges("plan_research", fan_out_research)
    builder.add_edge("research_worker", "synthesize_report")
    builder.add_edge("synthesize_report", "human_review")
    builder.add_conditional_edges("human_review", should_continue_review)
    builder.add_edge("save_report", END)
    return builder.compile(checkpointer=checkpointer, store=store)

# Exported graph for langgraph.json (platform provides checkpointer/store)
graph = build_graph()


# --- Interactive CLI ---

if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    memory = MemorySaver()
    store = InMemoryStore()
    app = build_graph(checkpointer=memory, store=store)

    def run_query(query: str):
        thread_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n[thread:{thread_id}] Researching: {query}\n")

        # Run graph — pauses at human_review
        for event in app.stream({"query": query}, config=config):
            for node in event:
                if node != "__interrupt__":
                    print(f"  [{node}] done")

        # Display report
        state = app.get_state(config)
        report = state.values.get("report", {})
        print(f"\n{'='*50}")
        print(f"Title: {report.get('title', 'N/A')}")
        print(f"Summary: {report.get('summary', 'N/A')}")
        for s in report.get("sections", []):
            print(f"\n## {s['heading']}")
            print(s["content"])
        print(f"\nSources ({len(report.get('sources', []))}):")
        for src in report.get("sources", []):
            print(f"  - {src['title']} {src.get('url', '')}")
        print(f"{'='*50}")

        # Human review loop
        while True:
            feedback = input("\nApprove? (yes / edit instructions): ").strip()
            if not feedback:
                continue
            if feedback.lower() in ("yes", "y", "approve"):
                for event in app.stream(Command(resume="approve"), config=config):
                    for node in event:
                        if node != "__interrupt__":
                            print(f"  [{node}] done")
                print("Report saved.")
                break
            else:
                for event in app.stream(Command(resume=feedback), config=config):
                    for node in event:
                        if node != "__interrupt__":
                            print(f"  [{node}] done")
                state = app.get_state(config)
                report = state.values.get("report", {})
                print(f"\n{'='*50}")
                print(f"Revised: {report.get('title', 'N/A')}")
                print(f"Summary: {report.get('summary', 'N/A')}")
                for s in report.get("sections", []):
                    print(f"\n## {s['heading']}")
                    print(s["content"])
                print(f"{'='*50}")

    def show_history():
        items = store.search(("research",))
        if not items:
            print("No past research.")
            return
        print(f"\nPast research ({len(items)}):")
        for item in items:
            print(f"  - {item.value['report'].get('title', item.value['query'])}")

    def show_time_travel(tid):
        config = {"configurable": {"thread_id": tid}}
        history = list(app.get_state_history(config))
        if not history:
            print("No history for this thread.")
            return
        print(f"\nCheckpoints ({len(history)}):")
        for i, s in enumerate(history):
            nxt = s.next[0] if s.next else "END"
            print(f"  [{i}] next={nxt} id={s.config['configurable']['checkpoint_id'][:8]}")

    # Main loop
    print("DeepLens Research Agent")
    print("Commands: /history | /travel <thread_id> | /quit")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        elif user_input == "/history":
            show_history()
        elif user_input.startswith("/travel"):
            parts = user_input.split(maxsplit=1)
            show_time_travel(parts[1]) if len(parts) > 1 else print("Usage: /travel <thread_id>")
        else:
            run_query(user_input)

    print("Goodbye.")
