from surya.common.progress import ProgressContext, ProgressEvent


def test_layout_returns_blocks(layout_predictor, test_image):
    events = []
    layout_results = layout_predictor([test_image], on_progress=events.append)
    context = ProgressContext(
        "balanced", "normal", "regions_and_reading_order", "pages"
    )
    assert events == [
        ProgressEvent("layout", 0, 1, context=context),
        ProgressEvent("layout", 1, 1, context=context),
    ]
    assert len(layout_results) == 1
    res = layout_results[0]
    assert res.image_bbox == [0, 0, 1024, 1024]
    if res.error:
        # Server may not be running in CI environments without llama-server
        return
    assert isinstance(res.bboxes, list)
    for box in res.bboxes:
        assert box.label
        assert box.count >= 0
        assert isinstance(box.position, int)
