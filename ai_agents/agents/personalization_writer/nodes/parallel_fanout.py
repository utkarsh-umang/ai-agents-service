import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from typing import Dict, List
from personalization_writer.state import AgentState


def flatten_dict(cleaned_urls: Dict):
    pairs =[]
    for cat,urls in cleaned_urls.items():
        for url in urls:
            pairs.append((cat,url))

    return pairs



async def scrape(cat, url, crawler,run_config):
    result = await crawler.arun(url=url, config=run_config)
    return (cat,result.markdown)


async def scraper(state: AgentState) -> AgentState:

    pairs = flatten_dict(state['cleaned_urls'])

    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig()   # Default crawl run configuration

    async with AsyncWebCrawler(config=browser_config) as crawler:

        tasks = [
            scrape(cat,url,crawler,run_config) for (cat,url) in pairs
        ]

        results = await asyncio.gather(*tasks)

    state['scraped_urls'] = {}

    for cat,content in results:
        if cat not in state['scraped_urls']:
            state['scraped_urls'][cat] = []   
        if state['scraped_urls'][cat] is not None:
            state['scraped_urls'][cat].append(content)
                
    return state

# if __name__ == "__main__":
#     asyncio.run(scraper())
