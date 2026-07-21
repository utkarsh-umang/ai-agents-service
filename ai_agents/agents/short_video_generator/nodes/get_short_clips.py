from dotenv import load_dotenv

import json

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langfuse.langchain import CallbackHandler

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

from tools.get_video_titles_tool import get_video_titles
from tools.get_short_clips_tool import get_shorts_clips

from contracts.models import VideoAnalysis

from helpers.validate_links import is_valid_youtube_url

load_dotenv()

# Langfuse
langfuse_handler = CallbackHandler()

# Tools
tools = [
    get_video_titles,
    get_shorts_clips
]

# LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash",
#     temperature=0.7
# )
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

llm_with_tools = llm.bind_tools(tools)

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

    response = llm_with_tools.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler]
        }
    )

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


def should_continue(state: MessagesState):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


graph = StateGraph(MessagesState)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue
)

graph.add_edge(
    "tools",
    "agent"
)

app = graph.compile()


def get_shorts(url: str) -> VideoAnalysis:
    
    if not is_valid_youtube_url(url):
        raise ValueError("URL not valid.")    

    query = f"""
Get shorts from this YouTube URL:
{url}
"""

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        },
        config={
            "callbacks": [langfuse_handler],
            "recursion_limit": 10
        }
    )

    for message in reversed(result["messages"]):

        if (
            isinstance(message, ToolMessage)
            and message.name == "get_shorts_clips"
        ):

            data = json.loads(message.content)

            return VideoAnalysis.model_validate(data)

    raise RuntimeError("get_shorts_clips did not return any result.")

