# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start <agent> --model <name>` picking a model from Settings > Connections.

Order: a model loaded in Unsloth, then a connection model (explicit
"connection:<id>/<model>" or a name exactly one opted-in connection lists), then a
local load. The agent gets the connection id, which the server routes to the connection.
"""

from __future__ import annotations

import pytest
import typer

import unsloth_cli.commands.start as start_cli


BASE = "http://127.0.0.1:8888"
KEY = "sk-unsloth-k"
RESIDENT = {"id": "unsloth/Qwen3-8B", "loaded": True}


def _connection(
    provider_id,
    name,
    models,
    *,
    allow = True,
):
    return {
        "id": provider_id,
        "display_name": name,
        "provider_type": "custom",
        "is_enabled": True,
        "models": models,
        "allow_api_key_access": allow,
    }


CUSTOM = _connection("3f9a1c2b7d4e8f60", "Custom", ["qwen3.8-flash-next-iq3_s", "qwen3:8b"])
CHATGPT = _connection("aa11bb22cc33dd44", "ChatGPT / Codex subscription", ["gpt-6-sol"])


class FakeServer:
    def __init__(
        self,
        connections,
        models = (RESIDENT,),
    ):
        self.connections = connections
        self.models = [dict(m) for m in models]
        self.loads = []
        self.requests = []

    def http_json(
        self,
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        self.requests.append((method, url))
        if url.endswith(("/api/inference/loaded-models", "/v1/models")):
            return {"data": [dict(m) for m in self.models]}
        if url.endswith("/api/providers/"):
            if isinstance(self.connections, Exception):
                raise self.connections
            return [dict(c) for c in self.connections]
        if url.endswith("/api/inference/status"):
            return {"is_gguf": False, "active_model": RESIDENT["id"]}
        raise AssertionError(f"unexpected request: {method} {url}")

    def load(self, base, key, requested, load, payload):
        self.loads.append(payload)
        self.models.append({"id": requested, "loaded": True})
        return {"status": "loaded", "model": requested}

    def install(self, monkeypatch, tmp_path):
        # --no-launch writes agent configs under the auth root; keep them out of the real ~/.unsloth.
        monkeypatch.setattr(start_cli, "_studio_auth_root", lambda: tmp_path / "auth")
        monkeypatch.setattr(start_cli, "_http_json", self.http_json)
        monkeypatch.setattr(start_cli, "_load_model_with_progress", self.load)
        monkeypatch.setattr(start_cli, "find_studio_server", lambda: BASE)
        monkeypatch.setattr(start_cli, "_agent_api_key", lambda *a, **k: KEY)
        return self


def _connect(model, load = start_cli.LoadOptions()):
    base, key, entry = start_cli._connect(None, model, load)
    return entry


def _failure(
    capsys,
    model,
    load = start_cli.LoadOptions(),
) -> str:
    with pytest.raises(typer.Exit):
        _connect(model, load)
    return capsys.readouterr().err


def test_a_unique_connection_model_becomes_its_connection_id(monkeypatch, tmp_path, capsys):
    server = FakeServer([CUSTOM, CHATGPT]).install(monkeypatch, tmp_path)

    entry = _connect("gpt-6-sol")

    assert entry["id"] == "connection:aa11bb22cc33dd44/gpt-6-sol"
    assert server.loads == []
    assert "from the ChatGPT / Codex subscription connection" in capsys.readouterr().out


def test_a_colon_in_a_connection_model_is_not_read_as_a_gguf_variant(monkeypatch, tmp_path):
    server = FakeServer([CUSTOM]).install(monkeypatch, tmp_path)

    entry = _connect("qwen3:8b")

    assert entry["id"] == "connection:3f9a1c2b7d4e8f60/qwen3:8b"
    assert server.loads == []


def test_a_loaded_local_model_wins_over_a_connection_model(monkeypatch, tmp_path):
    server = FakeServer([_connection("p1", "vLLM", [RESIDENT["id"]])]).install(
        monkeypatch, tmp_path
    )

    entry = _connect(RESIDENT["id"])

    assert entry["id"] == RESIDENT["id"]
    assert not any(url.endswith("/api/providers/") for _, url in server.requests)


def test_a_name_no_connection_lists_loads_locally(monkeypatch, tmp_path):
    server = FakeServer([CUSTOM]).install(monkeypatch, tmp_path)

    entry = _connect("unsloth/Qwen3-4B-GGUF")

    assert entry["id"] == "unsloth/Qwen3-4B-GGUF"
    assert server.loads == [{"model_path": "unsloth/Qwen3-4B-GGUF"}]


def test_an_unreadable_connection_list_falls_back_to_a_local_load(monkeypatch, tmp_path):
    server = FakeServer(OSError("refused")).install(monkeypatch, tmp_path)

    entry = _connect("unsloth/Qwen3-4B-GGUF")

    assert entry["id"] == "unsloth/Qwen3-4B-GGUF"
    assert len(server.loads) == 1


def test_an_explicit_connection_id_needs_no_model_list_entry(monkeypatch, tmp_path):
    FakeServer([CUSTOM]).install(monkeypatch, tmp_path)

    entry = _connect("connection:3f9a1c2b7d4e8f60/qwen3-next-80b")

    assert entry["id"] == "connection:3f9a1c2b7d4e8f60/qwen3-next-80b"


def test_an_explicit_id_for_an_unknown_connection_fails(monkeypatch, tmp_path, capsys):
    server = FakeServer([CUSTOM]).install(monkeypatch, tmp_path)

    assert "No enabled connection" in _failure(capsys, "connection:0000000000000000/x")
    assert server.loads == []


def test_a_model_on_two_connections_asks_for_the_full_id(monkeypatch, tmp_path, capsys):
    other = _connection("1234123412341234", "OpenRouter", ["gpt-6-sol"])
    FakeServer([CHATGPT, other]).install(monkeypatch, tmp_path)

    err = _failure(capsys, "gpt-6-sol")

    assert "--model connection:aa11bb22cc33dd44/gpt-6-sol" in err
    assert "--model connection:1234123412341234/gpt-6-sol" in err


def test_a_connection_without_the_opt_in_says_where_to_turn_it_on(monkeypatch, tmp_path, capsys):
    server = FakeServer([{**CHATGPT, "allow_api_key_access": False}]).install(monkeypatch, tmp_path)

    err = _failure(capsys, "gpt-6-sol")

    assert "Allow API keys and coding agents" in err
    assert server.loads == []


def test_an_older_server_without_the_opt_in_asks_for_an_update(monkeypatch, tmp_path, capsys):
    legacy = {k: v for k, v in CHATGPT.items() if k != "allow_api_key_access"}
    FakeServer([legacy]).install(monkeypatch, tmp_path)

    assert "Update Unsloth" in _failure(capsys, "gpt-6-sol")


def test_a_disabled_connection_is_ignored(monkeypatch, tmp_path):
    server = FakeServer([{**CHATGPT, "is_enabled": False}]).install(monkeypatch, tmp_path)

    _connect("gpt-6-sol")

    assert server.loads == [{"model_path": "gpt-6-sol"}]


def test_load_options_are_refused_for_a_connection_model(monkeypatch, tmp_path, capsys):
    FakeServer([CHATGPT]).install(monkeypatch, tmp_path)
    load = start_cli.LoadOptions(max_seq_length = 32768, supplied = frozenset({"max_seq_length"}))

    assert "do not apply to a connection model" in _failure(capsys, "gpt-6-sol", load)


def test_a_connection_id_never_auto_starts_a_server(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(start_cli, "find_studio_server", lambda: None)
    monkeypatch.setattr(
        start_cli,
        "_start_studio_server",
        lambda *a, **k: pytest.fail("a connection model cannot be served by `unsloth run`"),
    )

    with pytest.raises(typer.Exit):
        start_cli._require_studio("connection:aa11bb22cc33dd44/gpt-6-sol", serve = True)
    assert "No running Unsloth server" in capsys.readouterr().err


def test_downstream_gguf_and_variant_checks_skip_a_connection_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        start_cli, "_http_json", lambda *a, **k: pytest.fail("no status probe for a connection")
    )
    model_id = "connection:aa11bb22cc33dd44/gpt-6-sol"

    start_cli._require_gguf_for_agent(start_cli._CODEX_GGUF_AGENT, BASE, KEY, model_id)
    assert start_cli._subagent_model_id(BASE, KEY, {"id": model_id}, "gpt-6-sol", None) == model_id


@pytest.mark.parametrize("agent", ["claude", "codex", "opencode", "pi"])
def test_every_agent_is_configured_with_the_connection_id(monkeypatch, tmp_path, agent):
    from typer.testing import CliRunner

    FakeServer([CHATGPT]).install(monkeypatch, tmp_path)
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

    result = CliRunner().invoke(start_cli.start_app, [agent, "--no-launch", "--model", "gpt-6-sol"])

    assert result.exit_code == 0, f"{result.exception!r}\n{result.output}"
    assert "connection:aa11bb22cc33dd44/gpt-6-sol" in result.output
