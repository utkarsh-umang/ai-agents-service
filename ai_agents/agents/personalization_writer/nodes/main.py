from sitemap_generator import crawl_sitemap
from filter_suburls import categorize_suburls
from dfs import depth_calculation, sort_urls,chosen_urls
from parallel_fanout import scraper
from llm_scorer import create_prompt, llm_response

import asyncio

def main(website: str):
    suburls = asyncio.run(crawl_sitemap(website))
    cat_dict = categorize_suburls(suburls)

    cleaned_urls = chosen_urls(cat_dict)
    scraped_urls = asyncio.run(scraper(cleaned_urls))
    llm_responses = llm_response(scraped_urls)


    return llm_responses


if __name__ == "__main__":
    print(main("https://openai.com"))



