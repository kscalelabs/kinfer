"""Defines common types."""

__all__ = [
    "Metadata",
]

from pydantic import BaseModel


class Metadata(BaseModel):
    joint_names: list[str]
    num_commands: int | None
