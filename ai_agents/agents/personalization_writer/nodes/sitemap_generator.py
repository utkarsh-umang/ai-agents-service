from crawl4ai import AsyncUrlSeeder, SeedingConfig
import asyncio
from personalization_writer.state import AgentState


async def crawl_sitemap(state: AgentState) -> AgentState:

    state['suburls'] = []

    async with AsyncUrlSeeder() as seeder:
        config = SeedingConfig(source="sitemap+cc")
        urls = await seeder.urls(state['website'], config)

    for suburl in urls:
        state['suburls'].append(suburl['url'])

    return state


# if __name__ == "__main__":
#     results = asyncio.run(crawl_sitemap('https://openai.com'))
#     print(results)


