from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, get_ident

import pytest
from openai import OpenAI
from PIL import Image

from surya.inference.backends import openai_client
from surya.inference.schema import BatchInputItem, BatchOutputItem, GenerationResult


def test_progress_advances_before_first_item_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started: Event = Event()
    release_first: Event = Event()
    later_reported: Event = Event()
    progress: list[tuple[int, int]] = []
    callback_threads: list[int] = []
    consumer_threads: list[int] = []

    def generate(
        item: BatchInputItem,
        client: OpenAI,
        model_name: str,
        max_tokens_default: int,
        temperature: float,
        top_p: float,
        timeout: float,
        request_logprobs_default: bool,
    ) -> GenerationResult:
        index: int = item.metadata["index"]
        if index == 0:
            first_started.set()
            assert release_first.wait(5)
        else:
            assert first_started.wait(5)
        return GenerationResult(raw=str(index), token_count=1)

    def on_progress(completed: int, total: int) -> None:
        progress.append((completed, total))
        callback_threads.append(get_ident())
        later_reported.set()

    def run() -> list[BatchOutputItem]:
        consumer_threads.append(get_ident())
        return openai_client.chat_completions_batch(
            batch,
            client=OpenAI(api_key="test"),
            model_name="test",
            max_workers=2,
            on_progress=on_progress,
        )

    monkeypatch.setattr(openai_client, "_generate_one", generate)
    batch: list[BatchInputItem] = [
        BatchInputItem(Image.new("RGB", (1, 1)), "block", metadata={"index": index})
        for index in range(2)
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[list[BatchOutputItem]] = executor.submit(run)
        try:
            assert later_reported.wait(5)
            assert progress == [(1, 2)]
            assert not future.done()
        finally:
            release_first.set()
        results: list[BatchOutputItem] = future.result(timeout=5)
    assert [result.raw for result in results] == ["0", "1"]
    assert [result.metadata for result in results] == [{"index": 0}, {"index": 1}]
    assert progress == [(1, 2), (2, 2)]
    assert callback_threads == consumer_threads * 2


@pytest.mark.parametrize("final_error", [False, True])
@pytest.mark.parametrize("with_progress", [False, True])
def test_retry_completion_is_reported_once(
    monkeypatch: pytest.MonkeyPatch, final_error: bool, with_progress: bool
) -> None:
    attempts: list[BatchInputItem] = []
    progress: list[tuple[int, int]] = []

    def generate(
        item: BatchInputItem,
        client: OpenAI,
        model_name: str,
        max_tokens_default: int,
        temperature: float,
        top_p: float,
        timeout: float,
        request_logprobs_default: bool,
    ) -> GenerationResult:
        attempts.append(item)
        return GenerationResult(
            raw="" if len(attempts) == 1 or final_error else "recovered",
            token_count=0,
            error=len(attempts) == 1 or final_error,
        )

    def sleep(seconds: float) -> None:
        pass

    def on_progress(completed: int, total: int) -> None:
        progress.append((completed, total))

    monkeypatch.setattr(openai_client, "_generate_one", generate)
    monkeypatch.setattr(openai_client.time, "sleep", sleep)
    item: BatchInputItem = BatchInputItem(Image.new("RGB", (1, 1)), "block")
    results: list[BatchOutputItem] = openai_client.chat_completions_batch(
        [item],
        client=OpenAI(api_key="test"),
        model_name="test",
        max_retries=1,
        on_progress=on_progress if with_progress else None,
    )
    assert attempts == [item, item]
    assert progress == ([(1, 1)] if with_progress else [])
    assert results[0].error is final_error
    assert results[0].raw == ("" if final_error else "recovered")
