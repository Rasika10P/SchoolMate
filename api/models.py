"""Immutable curriculum domain objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Standard:
    framework_id: str
    code: str
    grade: int | None
    domain: str
    cluster: str
    text: str
    source_url: str
    page: int | None
    plain_summary: str | None = None
    plain_example: str | None = None


@dataclass(frozen=True)
class ProgressionStep:
    """One traversed edge; standard is the destination in the walk direction.

    Edge endpoints retain their stored orientation, including in backward walks.
    Depth is the hop at which the edge was first reached.
    """

    standard: Standard
    relation: str
    depth: int
    from_framework: str
    from_code: str
    to_framework: str
    to_code: str


@dataclass(frozen=True)
class AchievementDescriptor:
    framework_id: str
    grade: int
    subject: str
    level: int
    text: str
    source_url: str | None = None
    page: int | None = None


# Preserve the existing service-facing name.
AchievementLevel = AchievementDescriptor


class UnknownStandardError(LookupError):
    """A standard does not exist in the requested framework."""

    def __init__(self, framework_id: str, code: str) -> None:
        self.framework_id = framework_id
        self.code = code
        super().__init__(f"Unknown standard {code!r} in framework {framework_id!r}")
