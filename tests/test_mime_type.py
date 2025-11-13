from __future__ import annotations

from ebooktoc.siliconflow_api import _build_payload


def test_build_payload_uses_jpeg_mime_for_images():
    page_payloads = [
        {"page": 1, "image_b64": "AAA"},
    ]
    payload = _build_payload(page_payloads, max_pages=5)
    assert "messages" in payload
    messages = payload["messages"]
    assert isinstance(messages, list) and len(messages) >= 2
    user_content = messages[1]["content"]
    # Find the first image_url item
    image_items = [it for it in user_content if isinstance(it, dict) and "image_url" in it]
    assert image_items, "No image_url found in constructed payload"
    url = image_items[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,"), url

