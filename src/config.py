from pydantic import BaseModel, Field
from typing import Literal

"""
Configuration models used across the project.

This module defines typed, validated configuration objects for simulation
and related utilities. Instances are Pydantic models, so they perform
runtime validation and provide helpful error messages if inputs are invalid.
"""

class SimConfig(BaseModel):
    """
    Configuration for the telematics trip simulator.

    Attributes:
        hz: Samples per second emitted by the simulator.
        duration_sec: Total simulated trip duration, in seconds.
        mode: Driving behavior regime for the generator
              ("normal", "smooth", or "aggressive").
        out_path: Output path for the generated JSONL trip file.
    """
    hz: int = Field(10, description="Samples per second")
    duration_sec: int = Field(60, description="Total duration in seconds")
    mode: Literal["normal", "smooth", "aggressive"] = "normal"
    out_path: str = "data/samples/trip.jsonl"
