from pydantic import BaseModel, Field
from typing import Literal

class SimConfig(BaseModel):
    hz: int = Field(10, description="Samples per second")
    duration_sec: int = Field(60, description="Total duration in seconds")
    mode: Literal["normal", "smooth", "aggressive"] = "normal"
    out_path: str = "data/samples/trip.jsonl"
