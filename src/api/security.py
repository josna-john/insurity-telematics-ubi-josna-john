import os
from fastapi import Header, HTTPException

"""
Simple API key guard for FastAPI routes.

Reads the expected key from the environment variable `API_KEY` and compares it
to the inbound `X-API-Key` request header. If `API_KEY` is unset, the check is
considered disabled (all requests pass). If set and mismatched/missing, a 401
HTTPException is raised.
"""


def require_api_key(x_api_key: str | None = Header(default=None)):
    """
    Dependency for FastAPI endpoints to enforce an API key.

    Args:
        x_api_key (str | None): Value of the `X-API-Key` header, injected by FastAPI.

    Raises:
        HTTPException: 401 Unauthorized when the provided key does not match the
        `API_KEY` environment variable (or is missing when expected).
    """
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
