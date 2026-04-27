import asyncio
import json
import logging
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


DEFAULT_TIMEOUT = 7200  # 2 hours — batch runs with many instances need time


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    consumer: Consumer | None = None,
    a2a_client=None,
):
    """Returns dict with context_id, response and status (if exists)."""
    own_client = a2a_client is None
    if own_client:
        httpx_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        a2a_client = ClientFactory(config).create(agent_card)

    try:
        if consumer:
            await a2a_client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)

        last_event = None
        outputs: dict = {"response": "", "context_id": None}

        async for event in a2a_client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs
    finally:
        if own_client:
            await httpx_client.aclose()
