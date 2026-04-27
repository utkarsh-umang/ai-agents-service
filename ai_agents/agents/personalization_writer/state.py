from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    website: str
    suburls: List
    cat_dict: Dict
    cleaned_urls: Dict
    scraped_urls: Dict
    llm_responses: List

