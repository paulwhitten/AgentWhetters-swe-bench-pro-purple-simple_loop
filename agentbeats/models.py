from typing import Any
from pydantic import BaseModel, HttpUrl


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]
