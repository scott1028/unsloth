# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A saved connection's model, named "connection:<provider_id>/<model>", reached by API keys.

`unsloth start` hands coding agents that id. Claude Code (/v1/messages) and Codex
(/v1/responses) send only a model name, so the server has to route it to the saved
connection, and an sk-unsloth key may spend that connection's saved credential only
when the user opted the connection in.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import storage.providers_db as providers_db


CONNECTION_MODEL = "connection:saved-1/gpt-6-sol"


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _api_key_request(path: str = "/v1/chat/completions"):
    async def is_disconnected():
        return False

    return SimpleNamespace(
        headers = {"authorization": "Bearer sk-unsloth-test"},
        state = SimpleNamespace(skip_api_monitor = True),
        scope = {},
        url = SimpleNamespace(path = path),
        method = "POST",
        is_disconnected = is_disconnected,
    )


def _chat_sse(*chunks):
    async def gen():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type = "text/event-stream")


TEXT_THEN_TOOL = (
    {"choices": [{"index": 0, "delta": {"content": "Let me look."}}]},
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":'},
                        }
                    ]
                },
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]},
                "finish_reason": "tool_calls",
            }
        ]
    },
    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}},
)


@pytest.fixture()
def inf(monkeypatch):
    import auth.authentication as authentication
    import core.inference.llama_keepwarm as keepwarm
    from routes import inference

    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda _r: False)
    monkeypatch.setattr(keepwarm, "untrack_current_request", lambda _scope: None)
    monkeypatch.setattr(inference, "_request_is_saved_credential_workflow", lambda _r: False)
    return inference


@pytest.fixture()
def proxied(inf, monkeypatch):
    """Replace the external proxy; record the payload it was asked to send."""
    calls = []

    async def fake_proxy(
        payload,
        request,
        current_subject = None,
        **_kwargs,
    ):
        calls.append(payload)
        return _chat_sse(*TEXT_THEN_TOOL)

    monkeypatch.setattr(inf, "_proxy_to_external_provider", fake_proxy)
    return calls


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name and data is not None:
            events.append((name, data))
    return events


def _body(response) -> str:
    async def read():
        return "".join(
            [c.decode() if isinstance(c, bytes) else c async for c in response.body_iterator]
        )

    return _drive(read())


# ── id parsing ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model, binding",
    [
        ("connection:saved-1/gpt-6-sol", ("saved-1", "gpt-6-sol")),
        ("connection:abc/qwen3:8b", ("abc", "qwen3:8b")),
        ("connection:abc/openai/gpt-oss-20b", ("abc", "openai/gpt-oss-20b")),
        ("connection:abc", None),
        ("connection:/gpt", None),
        ("unsloth/Qwen3-8B", None),
        (None, None),
    ],
)
def test_connection_model_ids_parse_on_the_first_slash(inf, model, binding):
    assert inf._connection_model_binding(model) == binding


def test_an_explicit_provider_id_is_never_rebound(inf):
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        model = CONNECTION_MODEL,
        messages = [{"role": "user", "content": "hi"}],
        provider_id = "other",
    )
    assert inf._bind_connection_model(payload) is False
    assert payload.provider_id == "other"


# ── saved-credential policy ──────────────────────────────────────


class _FakeClient:
    def __init__(self, **kwargs):
        pass

    def stream_chat_completion(self, **kwargs):
        async def gen():
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return gen()

    async def close(self):
        return None


def _saved_connection(inf, monkeypatch, *, provider_type, allow_api_key_access):
    from core.inference.providers import get_base_url
    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": provider_type,
            "base_url": get_base_url(provider_type) or "http://127.0.0.1:8080/v1",
            "api_type": "chat_completions",
            "display_name": "Saved connection",
            "is_enabled": True,
            "allow_api_key_access": int(allow_api_key_access),
        },
    )
    monkeypatch.setattr(inf, "ExternalProviderClient", _FakeClient)


@pytest.mark.parametrize("allowed", [False, True])
def test_an_api_key_spends_a_saved_key_only_when_the_connection_opts_in(inf, monkeypatch, allowed):
    from models.inference import ChatCompletionRequest

    _saved_connection(inf, monkeypatch, provider_type = "openai", allow_api_key_access = allowed)
    seen = {}

    def resolve(
        provider_id,
        encrypted_api_key,
        *,
        allow_saved_key = True,
        **_kwargs,
    ):
        seen["allow_saved_key"] = allow_saved_key
        return "saved-key" if allow_saved_key else ""

    monkeypatch.setattr(inf, "resolve_provider_api_key_or_400", resolve)
    payload = ChatCompletionRequest(
        model = CONNECTION_MODEL,
        messages = [{"role": "user", "content": "hi"}],
        stream = True,
    )
    inf._bind_connection_model(payload)

    async def go():
        response = await inf._proxy_to_external_provider(payload, _api_key_request(), "t")
        return [chunk async for chunk in response.body_iterator]

    _drive(go())
    assert seen["allow_saved_key"] is allowed


def test_a_chatgpt_subscription_refuses_api_keys_until_opted_in(inf, monkeypatch):
    import core.inference.openai_codex_auth as codex_auth
    from models.inference import ChatCompletionRequest

    class PassedTheGate(Exception):
        pass

    def load_oauth_bundle(_provider_id):
        raise PassedTheGate()

    monkeypatch.setattr(codex_auth, "load_oauth_bundle", load_oauth_bundle)

    def run():
        payload = ChatCompletionRequest(
            model = CONNECTION_MODEL,
            messages = [{"role": "user", "content": "hi"}],
            stream = True,
        )
        inf._bind_connection_model(payload)
        return _drive(inf._proxy_to_external_provider(payload, _api_key_request(), "t"))

    _saved_connection(inf, monkeypatch, provider_type = "openai_codex", allow_api_key_access = False)
    with pytest.raises(HTTPException) as refused:
        run()
    assert refused.value.status_code == 403

    _saved_connection(inf, monkeypatch, provider_type = "openai_codex", allow_api_key_access = True)
    with pytest.raises(PassedTheGate):
        run()


# ── /v1/chat/completions ─────────────────────────────────────────


def test_chat_completions_routes_a_connection_model_before_any_local_switch(
    inf, proxied, monkeypatch
):
    from models.inference import ChatCompletionRequest

    async def no_switch(*_a, **_k):
        raise AssertionError("a connection model must not trigger a local model switch")

    monkeypatch.setattr(inf, "_maybe_auto_switch_model", no_switch)
    payload = ChatCompletionRequest(
        model = CONNECTION_MODEL,
        messages = [{"role": "user", "content": "hi"}],
        stream = True,
    )
    _drive(
        inf.produce_openai_chat_completions(
            payload, _api_key_request(), "t", cancel_on_disconnect = False
        )
    )
    assert proxied[0].provider_id == "saved-1"
    assert proxied[0].external_model == "gpt-6-sol"


def test_chat_completions_without_stream_returns_one_completion(inf, proxied):
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        model = CONNECTION_MODEL,
        messages = [{"role": "user", "content": "hi"}],
        stream = False,
    )
    response = _drive(
        inf.produce_openai_chat_completions(
            payload, _api_key_request(), "t", cancel_on_disconnect = False
        )
    )
    body = json.loads(response.body)

    # The proxy frames stream=false as SSE, so the route streams and collects instead.
    assert proxied[0].stream is True
    assert proxied[0].stream_options == {"include_usage": True}
    message = body["choices"][0]["message"]
    assert message["content"] == "Let me look."
    assert message["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path":"a.py"}',
    }
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["usage"]["completion_tokens"] == 7


# ── /v1/responses (Codex) ────────────────────────────────────────


def test_responses_without_stream_returns_one_response(inf, proxied):
    from models.inference import ResponsesRequest

    payload = ResponsesRequest(model = CONNECTION_MODEL, input = "open a.py")
    messages = inf._normalise_responses_input(payload)
    response = _drive(
        inf._responses_non_streaming(payload, messages, _api_key_request("/v1/responses"), "t")
    )
    body = json.loads(response.body)

    assert proxied[0].provider_id == "saved-1"
    assert [item["type"] for item in body["output"]] == ["message", "function_call"]
    assert body["output"][1]["arguments"] == '{"path":"a.py"}'
    assert body["usage"]["input_tokens"] == 12


def test_responses_stream_translates_the_connection_stream(inf, proxied):
    from models.inference import ResponsesRequest

    payload = ResponsesRequest(
        model = CONNECTION_MODEL,
        input = "open a.py",
        stream = True,
        tools = [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
    )
    messages = inf._normalise_responses_input(payload)
    response = _drive(
        inf._responses_stream(payload, messages, _api_key_request("/v1/responses"), None, "t")
    )
    events = _sse_events(_body(response))
    names = [name for name, _ in events]

    assert proxied[0].provider_id == "saved-1"
    assert proxied[0].stream is True
    assert "response.output_text.delta" in names
    added = [d["item"] for n, d in events if n == "response.output_item.added"]
    calls = [item for item in added if item["type"] == "function_call"]
    assert calls[0]["name"] == "read_file"
    done = [d["item"] for n, d in events if n == "response.output_item.done"]
    assert any(item.get("arguments") == '{"path":"a.py"}' for item in done)
    terminal_name, terminal = events[-1]
    assert terminal_name == "response.completed"
    assert terminal["response"]["model"] == CONNECTION_MODEL
    assert terminal["response"]["usage"]["input_tokens"] == 12


# ── /v1/messages (Claude Code) ───────────────────────────────────


def _anthropic_payload(**overrides):
    from models.inference import AnthropicMessagesRequest

    base = dict(
        model = CONNECTION_MODEL,
        max_tokens = 256,
        messages = [{"role": "user", "content": "open a.py"}],
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ],
    )
    base.update(overrides)
    return AnthropicMessagesRequest(**base)


def test_messages_stream_translates_the_connection_stream(inf, proxied):
    response = _drive(
        inf.anthropic_messages(
            _anthropic_payload(stream = True), _api_key_request("/v1/messages"), "t"
        )
    )
    events = _sse_events(_body(response))
    names = [name for name, _ in events]

    sent = proxied[0]
    assert sent.provider_id == "saved-1" and sent.external_model == "gpt-6-sol"
    assert sent.tools[0]["function"]["name"] == "read_file"
    assert names[0] == "message_start" and names[-1] == "message_stop"
    starts = [d["content_block"] for n, d in events if n == "content_block_start"]
    assert [block["type"] for block in starts] == ["text", "tool_use"]
    assert starts[1]["name"] == "read_file"
    stop = next(d for n, d in events if n == "message_delta")
    assert stop["delta"]["stop_reason"] == "tool_use"


def test_messages_without_stream_returns_one_message(inf, proxied):
    response = _drive(
        inf.anthropic_messages(_anthropic_payload(), _api_key_request("/v1/messages"), "t")
    )
    body = json.loads(response.body)

    assert [block["type"] for block in body["content"]] == ["text", "tool_use"]
    assert body["content"][1]["input"] == {"path": "a.py"}
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["input_tokens"] == 12


def test_messages_refuse_unsloth_server_tools_for_a_connection_model(inf, proxied):
    payload = _anthropic_payload(tools = None, enable_tools = True)
    with pytest.raises(HTTPException) as refused:
        _drive(inf.anthropic_messages(payload, _api_key_request("/v1/messages"), "t"))
    assert refused.value.status_code == 400
    assert proxied == []


def test_count_tokens_estimates_without_loading_a_model(inf, monkeypatch):
    async def no_switch(*_a, **_k):
        raise AssertionError("a connection model must not trigger a local model switch")

    monkeypatch.setattr(inf, "_maybe_auto_switch_model", no_switch)
    counted = _drive(
        inf.anthropic_count_tokens(
            _anthropic_payload(), _api_key_request("/v1/messages/count_tokens"), "t"
        )
    )
    assert counted["input_tokens"] > 0


# ── the opt-in column ────────────────────────────────────────────


@pytest.fixture()
def isolated_providers_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db_path)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    providers_db._schema_ready = set()
    yield db_path
    providers_db._schema_ready = set()


def test_the_opt_in_defaults_off_and_round_trips(isolated_providers_db: Path):
    providers_db.create_provider(
        id = "custom1",
        provider_type = "custom",
        display_name = "Custom",
        base_url = "http://127.0.0.1:8080/v1",
    )
    assert not providers_db.get_provider("custom1")["allow_api_key_access"]

    providers_db.update_provider(id = "custom1", allow_api_key_access = True)
    assert providers_db.get_provider("custom1")["allow_api_key_access"]

    # An unrelated edit leaves the opt-in alone.
    providers_db.update_provider(id = "custom1", display_name = "Renamed")
    assert providers_db.get_provider("custom1")["allow_api_key_access"]

    providers_db.update_provider(id = "custom1", allow_api_key_access = False)
    assert not providers_db.get_provider("custom1")["allow_api_key_access"]
