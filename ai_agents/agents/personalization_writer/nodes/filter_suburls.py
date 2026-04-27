from personalization_writer.nodes.sitemap_generator import crawl_sitemap
from typing import List
import asyncio
from personalization_writer.state import AgentState


CATEGORY_KEYWORDS = {
	"ABOUT_US": [
		"about", "who-we-are", "company", "our-story", "mission", "values", 
		"about-us", "story", "timeline", "milestones", "why-us"
	],
	
	"EBOOK": [
		"ebook", "e-book", "whitepaper", "white-paper", "guide", "pdf", 
		"resources", "downloads", "books", "library", "documents"
	],
	
	"COURSES": [
		"course", "academy", "learning", "training", "workshop",
		"certification", "program", "bootcamp", "masterclass", 
		"education", "class", "e-learning"
	],
	
	"RECENT_BLOG": [
		"blog", "insights", "articles", "news", "updates", 
		"post", "media", "latest", "trends", "press", 
		"content-hub"
	],
	
	"TESTIMONIALS": [
		"testimonial", "reviews", "case-study", "success-story", 
		"client-story", "customer-story", "feedback", "clients", 
		"portfolio", "results", "social-proof"
	],
	
	"WEBINAR": [
		"webinar", "event", "session", "live", "virtual-event", 
		"presentation", "conference", "summit", "registration", 
		"upcoming", "schedule"
	],
	
	"SERVICES": [
		"service", "solution", "offering", "expertise", "consulting", 
		"what-we-do", "capability", "support", "practice", 
		"professional-services", "how-we-help"
	],
	
	"PODCAST": [
		"podcast", "episodes", "audio", "listen", "show", 
		"interview", "series", "stream", "speakers", 
		"voice", "subscribe"
	],
	
	"SHOP": [
		"shop", "store", "buy", "purchase", "products", 
		"cart", "checkout", "pricing", "e-commerce", 
		"merchandise", "order"
	]
}

def categorize_suburls(state: AgentState) -> AgentState:

    state['cat_dict'] = {key: [] for key in CATEGORY_KEYWORDS}

    for suburl in state['suburls']:
        for key, value in CATEGORY_KEYWORDS.items():
            if any(k in suburl.lower() for k in value):
                state['cat_dict'][key].append(suburl)

    return state



# if __name__ == "__main__":
#     categorize_suburls()