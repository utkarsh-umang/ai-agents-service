from personalization_writer.nodes.filter_suburls import categorize_suburls
from typing import Dict
import re
from urllib.parse import urlparse
from personalization_writer.state import AgentState


CATEGORY_RULES = {
	'ABOUT_US': 'ascending',
	'EBOOK': 'ascending',
	'COURSES': 'ascending',
	'RECENT_BLOG': 'descending',
	'TESTIMONIALS': 'ascending',
	'WEBINAR': 'descending',
	'SERVICES': 'descending',
	'PODCAST': 'descending',
	'SHOP': 'ascending'
}


def depth_calculation(category_dict: Dict):

    url_depth = {}

    for key, values in category_dict.items():
        url_depth[key] = []

        for url in values:
            path = urlparse(url).path
            depth = (len(path.strip('/').split('/')))

            url_depth[key].append((url,depth))

    return url_depth


def sort_urls(url_depth: Dict):
    
    for cat,sort_order in CATEGORY_RULES.items():
        if cat in url_depth.keys():
            if sort_order == 'ascending':
                url_depth[cat].sort(key = lambda x: x[1],reverse=False)
            else:
                url_depth[cat].sort(key = lambda x: x[1],reverse=True)
        else:
            print("category not found")

    return url_depth


def chosen_urls(state: AgentState) -> AgentState:

    depth_urls = depth_calculation(state['cat_dict'])

    sorted_urls = sort_urls(depth_urls)
    
    state['cleaned_urls'] = {}
    for cat,suburls in sorted_urls.items():
        state['cleaned_urls'][cat] = []

        if len(suburls) > 10:
            chosen_suburls = suburls[:10]
        else:
            chosen_suburls = suburls[:]


        for surl, depth in chosen_suburls:
            state['cleaned_urls'][cat].append(surl)


    return state
        


# if __name__ == "__main__":

#     cat_dict = categorize_suburls
#     result = depth_calculation()