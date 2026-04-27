from personalization_writer.nodes.sitemap_generator import crawl_sitemap
from personalization_writer.nodes.filter_suburls import categorize_suburls
from personalization_writer.nodes.dfs import chosen_urls
from personalization_writer.nodes.parallel_fanout import scraper
from personalization_writer.nodes.llm_scorer import llm_response

import asyncio
from langgraph.graph import StateGraph, START, END
from personalization_writer.state import AgentState
from PIL import Image, ImageShow

graph = StateGraph(AgentState)

graph.add_node("sitemap_crawler_node",crawl_sitemap)
graph.add_node("categorize_suburls_node",categorize_suburls)
graph.add_node("dfs_urls_node",chosen_urls)
graph.add_node("scraper_node",scraper)
graph.add_node("llm_response_node",llm_response)

graph.add_edge(START, "sitemap_crawler_node")
graph.add_edge("sitemap_crawler_node", "categorize_suburls_node")
graph.add_edge("categorize_suburls_node", "dfs_urls_node")
graph.add_edge("dfs_urls_node", "scraper_node")
graph.add_edge("scraper_node", "llm_response_node")
graph.add_edge("llm_response_node", END)

agent = graph.compile()

agent.get_graph().draw_mermaid_png(output_file_path="graph.png")

async def main():
    result = await agent.ainvoke({"website": "https://openai.com"})
    print(result)

asyncio.run(main())
