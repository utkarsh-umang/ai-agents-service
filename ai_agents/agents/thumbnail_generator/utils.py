import httpx


_DEFAULT_HEADERS = {
    "User-Agent": "ai-agents-service/0.1 (thumbnail-generator; httpx)",
}

def fetch_and_encode(url: str) -> tuple[bytes, str]:
    """Fetch image from URL; return (raw_bytes, mime_type)."""
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=60.0,
        headers=_DEFAULT_HEADERS,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/jpeg")
    mime_type = content_type.split(";")[0].strip()
    return response.content, mime_type