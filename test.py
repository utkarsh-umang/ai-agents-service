import asyncio
from ai_agents.agents.email_finder.io.contract_models import CrawlPageInput
from ai_agents.agents.email_finder.state import CanonicalLead, Identity
from ai_agents.agents.email_finder.nodes.crawl_page import _crawl_async

lead = CanonicalLead(
    identity=Identity(host_name="Patrick"),
    website="https://httpbin.org/html",  # simple open test page
)

inp = CrawlPageInput(
    url="https://httpbin.org/html",
    lead=lead,
    trace_id="test-001",
    page_timeout_ms=60000,
)

result = asyncio.run(_crawl_async(inp))
print("=== CRAWL SUCCESS:", result)
print("=== CANDIDATES:", result.candidates)
print("=== ERRORS:", result.errors)

for c in result.candidates:
    print(c.email, "|", c.confidence, "|", c.note)


print("=== REAL ENRICHMENT ===")



from ai_agents.agents.email_finder.nodes.crawl_page import (
    _confidence_for_email,
    _enrich_confidence,   # whatever you named the new enrichment function
)

# Simulate what crawl_page does after fetching
fake_url = "https://podcasters.spotify.com/pod/show/josca-s-moore7"
fake_blob = "Welcome to the show hosted by Dustin. Contact us at dustin5706@hotmail.com"

test_emails = ["dustin5706@hotmail.com", "josca@goongym.com", "info@site.com"]

for email in test_emails:
    base = _confidence_for_email(email)
    enriched_conf, note = _enrich_confidence(email, fake_url, fake_blob, base)
    print(f"{email} | base={base} | enriched={enriched_conf} | note={note}")



import asyncio
from ai_agents.agents.email_finder.nodes.validate_existing_email import (
    validate_existing_email_node_async,
)

# Test 1: should PASS — "patrick" likely on chasing excellence page
state_pass = {
    "lead": {
        "identity": {"host_name": "Patrick"},
        "website": "https://art19.com/shows/chasing-excellence",
        "existing_email": "patrick@functionalprojects.com",
        "existing_email": "patrick@functionalprojects.com",
        "social_links": {},
        "discovery_urls": [],
        "source_type": "other",
        "raw": {}
    },
    "trace_id": "test-validate-001"
}

# Test 2: should FAIL threshold if stricter — no name match
state_fail = {
    "lead": {
        "identity": {"host_name": "John"},
        "website": "https://httpbin.org/html",
        "existing_email": "xyz1234@randomsite.com",
        "social_links": {},
        "discovery_urls": [],
        "source_type": "other",
        "raw": {}
    },
    "trace_id": "test-validate-002"
}

print("=== TEST 1 (expect found or not_found depending on page) ===")
print(asyncio.run(validate_existing_email_node_async(state_pass)))

print("=== TEST 2 (expect not_found if threshold is strict) ===")
print(asyncio.run(validate_existing_email_node_async(state_fail)))


from ai_agents.agents.email_finder.nodes.crawl_page import _extract_fb_links

# Test FB extraction
html = '''
<a href="https://www.facebook.com/chasingexcellencepodcast">Follow us</a>
<a href="https://www.facebook.com/sharer/sharer.php?u=test">Share</a>
<a href="https://www.facebook.com/dialog/feed">Dialog</a>
<a href="https://twitter.com/someone">Twitter</a>
'''

links = _extract_fb_links(html)
print("=== FB LINKS ===", links)
# Expected: only ["https://www.facebook.com/chasingexcellencepodcast"]
# sharer, dialog should be filtered out
# twitter should not appear


from ai_agents.agents.email_finder.graph import build_graph
g = build_graph()
print("=== GRAPH COMPILED OK ===")


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