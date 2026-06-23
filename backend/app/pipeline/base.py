from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.storage.artifacts import ArtifactStore
from app.config.settings import settings


@dataclass
class StageConfig:
    gpu_enabled: bool = True
    processing_fps: int = settings.processing_fps
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    status: str
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(cls, artifacts: dict[str, Path] | None = None, metadata: dict[str, Any] | None = None) -> "StageResult":
        return cls(status="success", artifacts=artifacts or {}, metadata=metadata or {})

    @classmethod
    def from_error(cls, message: str) -> "StageResult":
        return cls(status="error", error=message)

    @classmethod
    def skipped(cls, reason: str = "") -> "StageResult":
        return cls(status="skipped", metadata={"reason": reason})


class PipelineStage(Protocol):
    name: str
    input_keys: list[str]
    output_keys: list[str]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult: ...
