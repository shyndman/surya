from collections.abc import Callable

import httpx
import pytest

from surya.inference.backends import spawn

MODEL_NAME = "datalab-to/surya-ocr-2"
OTHER_MODEL_NAME = "other-model"
ROUTER_URL = "http://router.test/v1"


@pytest.fixture
def serve_models(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str]], None]:
    client_type = httpx.Client

    def serve(model_names: list[str]) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200)
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": name} for name in model_names]})

        def client(*, timeout: float) -> httpx.Client:
            return client_type(timeout=timeout, transport=httpx.MockTransport(respond))

        monkeypatch.setattr(spawn.httpx, "Client", client)

    return serve


def attach() -> spawn.SpawnedServer:
    def do_not_spawn(port: int) -> spawn.SpawnHandle:
        raise AssertionError("An external router must not start a server")

    return spawn.attach_or_spawn(
        backend="llamacpp",
        expected_model_name=MODEL_NAME,
        spawn_fn=do_not_spawn,
        health_url_for=lambda port: f"http://localhost:{port}",
        openai_url_for=lambda port: f"http://localhost:{port}/v1",
        external_url=ROUTER_URL,
    )


@pytest.mark.parametrize(
    "model_names",
    [[MODEL_NAME], [OTHER_MODEL_NAME, MODEL_NAME], [MODEL_NAME, OTHER_MODEL_NAME]],
)
def test_attachment_selects_requested_model(
    serve_models: Callable[[list[str]], None], model_names: list[str]
) -> None:
    serve_models(model_names)
    handle = attach()
    assert handle.model_name == MODEL_NAME
    assert handle.base_url == ROUTER_URL
    assert not handle.spawned_by_us


def test_attachment_rejects_missing_model(
    serve_models: Callable[[list[str]], None],
) -> None:
    serve_models([OTHER_MODEL_NAME])
    with pytest.raises(spawn.SpawnError, match="Model mismatch"):
        attach()
