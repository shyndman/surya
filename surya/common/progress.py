from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

LayoutPurpose = Literal["normal", "ocr_fallback"]
OCRPurpose = Literal["normal", "flagged_blocks", "table", "equation"]
OCRPhase = Literal["recognition", "regeneration", "block_fallback"]


@dataclass(frozen=True)
class ProgressContext:
    mode: Literal["balanced", "fast", "full_page", "block"]
    purpose: LayoutPurpose | OCRPurpose
    phase: Literal["regions", "regions_and_reading_order"] | OCRPhase
    unit: Literal["pages", "blocks"]
    attempt: int = 1


@dataclass(frozen=True)
class ProgressEvent:
    operation: Literal["detection", "ocr_error", "layout", "ocr"]
    completed: int
    total: int
    context: ProgressContext | None = None


ProgressHandler = Callable[[ProgressEvent], None]
BatchProgressHandler = Callable[[int, int], None]
