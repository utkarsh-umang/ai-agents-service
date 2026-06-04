import asyncio
from ai_agents.agents.email_finder.io.contract_models import CrawlPageInput
from ai_agents.agents.email_finder.state import CanonicalLead, Identity
from ai_agents.agents.email_finder.nodes.crawl_page import _crawl_async

lead = CanonicalLead(
    identity=Identity(host_name="Dustin"),
    website="https://sites.libsyn.com/609600",
)

inp = CrawlPageInput(
    url="https://sites.libsyn.com/609600",
    lead=lead,
    trace_id="test-001",
    page_timeout_ms=60000,
)

result = asyncio.run(_crawl_async(inp))

for c in result.candidates:
    print(c.email, "|", c.confidence, "|", c.note)



# Add RAPIDAPI_KEY to your .env first, then:

import asyncio
from ai_agents.agents.email_finder.nodes.fb_crawler import fb_crawler_node_async

state = {
    "lead": {
        "identity": {"host_name": "Western Sky"},
        "website": "westernskyenergy.com",
        "existing_email": None,
        "social_links": {"facebook": "https://www.facebook.com/westernskyenergy"},
        "discovery_urls": [],
        "source_type": "other",
        "raw": {}
    },
    "scraped_fb_links": [],
    "trace_id": "test-fb-001"
}

print(asyncio.run(fb_crawler_node_async(state)))
# Expected: email_found, careers@westernskyenergy.com

from ai_agents.agents.email_finder.graph import build_graph
g = build_graph()
print("=== GRAPH COMPILED OK ===")