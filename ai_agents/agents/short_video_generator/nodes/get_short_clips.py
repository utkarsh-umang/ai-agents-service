import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai_agents.agents.short_video_generator.contracts import VideoAnalysis
from ai_agents.agents.short_video_generator.helpers.validate_links import (
    is_valid_youtube_url,
)
from ai_agents.agents.short_video_generator.tools.get_short_clips_tool import (
    get_shorts_clips,
)
from ai_agents.agents.short_video_generator.tools.get_video_titles_tool import (
    get_video_titles,
)

logger = logging.getLogger(__name__)

ROUTER_MODEL = "gpt-4o-mini"
# This model only routes to tools; its prose is discarded (see run_short_video_agent).
ROUTER_TEMPERATURE = 0.0
RECURSION_LIMIT = 10

TOOLS = [get_video_titles, get_shorts_clips]

_llm_with_tools = None
_langfuse_handler = None
_app = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = ChatOpenAI(
            model=ROUTER_MODEL, temperature=ROUTER_TEMPERATURE
        ).bind_tools(TOOLS)
    return _llm_with_tools


def _get_callbacks() -> list:
    """Langfuse callback, if Langfuse is configured.

    Returns an empty list when the handler cannot be constructed, so tracing
    being unavailable degrades observability instead of failing the job. This
    used to be built at import time, which made Langfuse config a hard
    requirement for importing the agent at all.
    """
    global _langfuse_handler
    if _langfuse_handler is None:
        try:
            from langfuse.langchain import CallbackHandler

            _langfuse_handler = CallbackHandler()
        except Exception as exc:
            logger.warning("langfuse tracing disabled: %s", exc)
            _langfuse_handler = False
    return [_langfuse_handler] if _langfuse_handler else []

SYSTEM_PROMPT = """
You are a YouTube AI assistant.

Available tools:
- get_video_titles
- get_shorts_clips

Instructions:

1. Decide which tools are needed.
2. You may call multiple tools.
3. Always use tool outputs. Never invent information.
4. Do not summarize, explain, or rewrite tool outputs in natural language.
5. Your final response must be ONLY a valid JSON object. Do not include markdown, code fences, headings, or additional text.
6. If the URL is not a valid YouTube channel or video URL, return:

{{
  "error": "Url is not valid"
}}

7. If the requested task cannot be completed using the available tools, return:

{{
  "error": "Cannot perform given task"
}}
8. If get_shorts_clips has been called, use its output for making final response. Do not add or remove clips, modify timestamps, rewrite explanations, or generate your own summary.
Return  ONLY JSON using data from  get_shorts_clips tool using the following schema:

{{
    "video_url": "Url of the youtube video",
    {{
        "clips": [
            {{
                "start_timestamp": "HH:MM:SS",
                "end_timestamp": "HH:MM:SS",
                "duration_seconds": <integer_between_60_and_80>,
                "title": "Short catchy title",
                "topic": "What this clip talks about",
                "why_it_works": "Why this segment would perform well as a short-form video"
            }}
        ]
    }}
}}
"""


def agent(state: MessagesState):

    messages = state["messages"]

    # Add system prompt only once
    if len(messages) == 1:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages
        ]

    response = _get_llm_with_tools().invoke(
        messages,
        config={"callbacks": _get_callbacks()},
    )

    return {
        "messages": [response]
    }


def should_continue(state: MessagesState):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


def _get_app():
    """Compile the graph on first use and cache it.

    Compiling at import time meant importing this module built a LangGraph and
    an LLM client as a side effect.
    """
    global _app
    if _app is None:
        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode(TOOLS))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")
        _app = graph.compile()
    return _app


def run_short_video_agent(video_url: str) -> VideoAnalysis:
    """Analyse a long-form YouTube video and return clip candidates.

    Returns timestamps and reasoning only — never video bytes. Downloading and
    cutting are the caller's job (see the helpers package).

    Args:
        video_url: A YouTube watch URL.

    Returns:
        VideoAnalysis with 5-10 clips of 60-80s, each carrying a title, topic
        and why_it_works.

    Raises:
        ValueError: if ``video_url`` is not a valid YouTube URL.
        RuntimeError: if the agent never called the clip-selection tool.
    """
    if not is_valid_youtube_url(video_url):
        raise ValueError(f"not a valid YouTube URL: {video_url!r}")

    query = f"Get shorts from this YouTube URL:\n{video_url}"

    result = _get_app().invoke(
        {"messages": [HumanMessage(content=query)]},
        config={
            "callbacks": _get_callbacks(),
            "recursion_limit": RECURSION_LIMIT,
        },
    )

    # Read the tool's own output rather than the model's final message: the
    # router is free to paraphrase, and the tool already returns a validated
    # VideoAnalysis. Walk backwards to pick up the most recent call.
    for message in reversed(result["messages"]):
        if isinstance(message, ToolMessage) and message.name == "get_shorts_clips":
            return VideoAnalysis.model_validate(json.loads(message.content))

    raise RuntimeError(
        "the agent finished without calling get_shorts_clips — no clips produced"
    )


# Original name, kept so the notebook keeps working.
get_shorts = run_short_video_agent

