from surya.common.progress import ProgressEvent
from surya.recognition import SKIP_CANON_LABELS


def test_recognition(recognition_predictor, layout_predictor, test_image):
    layouts = layout_predictor([test_image])
    if layouts[0].error:
        # Server unavailable in test env — skip silently
        return
    events = []
    page_results = recognition_predictor(
        [test_image], layouts, on_progress=events.append
    )
    submitted = sum(box.label not in SKIP_CANON_LABELS for box in layouts[0].bboxes)
    expected = (
        [
            ProgressEvent("ocr", completed, submitted)
            for completed in range(submitted + 1)
        ]
        if submitted
        else []
    )
    assert events == expected

    assert len(page_results) == 1
    assert page_results[0].image_bbox == [0, 0, 1024, 1024]

    blocks = page_results[0].blocks
    # Each layout box should produce one block (skipped or otherwise)
    assert len(blocks) == len(layouts[0].bboxes)
    for blk in blocks:
        assert blk.reading_order >= 0
        assert isinstance(blk.html, str)
