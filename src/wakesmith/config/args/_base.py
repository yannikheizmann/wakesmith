"""Define the strict base class shared by all argument groups."""

from pydantic import BaseModel, ConfigDict


class ArgsBase(BaseModel):
    """Argument group that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
