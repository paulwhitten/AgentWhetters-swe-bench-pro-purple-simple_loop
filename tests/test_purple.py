"""Smoke tests for the SWE-bench purple agent server."""

import json
import pytest
import httpx


@pytest.mark.asyncio
async def test_agent_card(purple_url):
    """The agent card endpoint should return valid JSON with expected fields."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{purple_url}/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "AgentWhetters_SWEBench"
    assert len(card["skills"]) >= 1
