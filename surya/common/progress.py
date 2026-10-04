from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProgressEvent:
    operation: Literal["detection", "ocr_error"]
    completed: int
    total: int


ProgressHandler = Callable[[ProgressEvent], None]
