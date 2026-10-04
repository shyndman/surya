import pytest
from PIL import Image

from surya.common.progress import BatchProgressHandler, ProgressEvent
from surya.inference import SuryaInferenceManager
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import PageOCRResult


class ProgressManager(SuryaInferenceManager):
    def __init__(self) -> None:
        self.submitted: list[int] = []

    def generate(
        self,
        batch: list[BatchInputItem],
        *,
        on_progress: BatchProgressHandler | None = None,
    ) -> list[BatchOutputItem]:
        self.submitted.append(len(batch))
        outputs: list[BatchOutputItem] = []
        for completed, item in enumerate(batch, 1):
            outputs.append(
                BatchOutputItem(
                    raw="<p>text</p>" if item.prompt_type == "block" else "",
                    token_count=1,
                    error=item.prompt_type != "block",
                    metadata=item.metadata,
                )
            )
            if on_progress is not None:
                on_progress(completed, len(batch))
        return outputs


def make_layout() -> LayoutResult:
    return LayoutResult(
        bboxes=[
            LayoutBox(
                polygon=[0, 0, 10, 10], label=label, raw_label=label, position=index
            )
            for index, label in enumerate(["Text", "Picture"])
        ],
        image_bbox=[0, 0, 10, 10],
    )


def test_layout_initial_progress_precedes_manager_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[ProgressEvent] = []
    manager: ProgressManager = ProgressManager()

    def get_manager() -> SuryaInferenceManager:
        assert events == [ProgressEvent("layout", 0, 2)]
        return manager

    monkeypatch.setattr("surya.layout.get_default_manager", get_manager)
    LayoutPredictor()([Image.new("RGB", (10, 10))] * 2, on_progress=events.append)
    assert events == [ProgressEvent("layout", count, 2) for count in range(3)]


def test_recognition_regeneration_batches_and_fallback_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("surya.recognition.settings.SURYA_FULLPAGE_REGEN", True)
    monkeypatch.setattr("surya.recognition._REGEN_ROUNDS", [(0.0, None), (0.2, 0.95)])
    manager: ProgressManager = ProgressManager()
    events: list[ProgressEvent] = []
    images: list[Image.Image] = [Image.new("RGB", (10, 10), "black")] * 2
    RecognitionPredictor(manager)(
        images, [make_layout()] * 2, full_page=True, on_progress=events.append
    )
    assert manager.submitted == [2, 2, 2]
    assert events == [ProgressEvent("ocr", count, 2) for count in range(3)] * 3


def test_recognition_initial_progress_precedes_manager_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[ProgressEvent] = []
    manager: ProgressManager = ProgressManager()

    def get_manager() -> SuryaInferenceManager:
        assert events == [ProgressEvent("ocr", 0, 1)]
        return manager

    monkeypatch.setattr("surya.recognition.get_default_manager", get_manager)
    RecognitionPredictor()(
        [Image.new("RGB", (10, 10))], [make_layout()], on_progress=events.append
    )
    assert manager.submitted == [1]
    assert events == [ProgressEvent("ocr", 0, 1), ProgressEvent("ocr", 1, 1)]


def test_recognition_skipped_only_batch_has_no_progress() -> None:
    events: list[ProgressEvent] = []
    manager: ProgressManager = ProgressManager()
    layout: LayoutResult = make_layout()
    layout.bboxes = [layout.bboxes[1]]
    results: list[PageOCRResult] = RecognitionPredictor(manager)(
        [Image.new("RGB", (10, 10))], [layout], on_progress=events.append
    )
    assert manager.submitted == []
    assert events == []
    assert results[0].blocks[0].skipped
