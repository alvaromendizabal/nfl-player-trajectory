"""Exercise OAuth entry, saved-session reuse, and safe failure reporting offline."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest


@pytest.fixture
def auth(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/authenticate.py"
    spec = importlib.util.spec_from_file_location("authenticate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(module, "KaggleClient", MagicMock(return_value=client))
    credentials = MagicMock()
    monkeypatch.setattr(module, "KaggleCredentials", credentials)
    oauth = MagicMock()
    monkeypatch.setattr(module, "KaggleOAuth", oauth)
    return module, credentials, oauth


def events(root):
    files = list((root / "logs").glob("*-authenticate.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


@pytest.mark.parametrize("state", ["fresh", "reuse", "force"])
def test_oauth_entry_and_saved_session_reuse(auth, tmp_path, state):
    module, credentials, oauth = auth
    saved = None if state == "fresh" else MagicMock()
    credentials.load.return_value = saved
    assert module.authenticate(tmp_path, force=state == "force") == 0
    if state == "reuse":
        oauth.assert_not_called()
        assert saved.mock_calls == [call.get_access_token(), call.introspect()]
    else:
        oauth.return_value.authenticate.assert_called_once_with(
            scopes=["resources.admin:*"], no_launch_browser=True
        )
        if saved is not None:
            assert saved.mock_calls == []
    records = events(tmp_path)
    assert records[0]["event"] == "started"
    assert records[-1]["event"] == "completed"
    assert all(record["timestamp"].endswith("+00:00") for record in records)
    assert records[-1]["elapsed_seconds"] >= records[0]["elapsed_seconds"]
    assert not (tmp_path / ".kaggle").exists()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [(RuntimeError("private-session-value"), 1), (EOFError(), 130), (KeyboardInterrupt(), 130)],
)
def test_failed_sign_in_never_reports_success_or_logs_secrets(
    auth, tmp_path, capsys, failure, expected
):
    module, credentials, oauth = auth
    credentials.load.return_value = None
    oauth.return_value.authenticate.side_effect = failure
    assert module.authenticate(tmp_path) == expected
    records = events(tmp_path)
    output = capsys.readouterr().out
    assert records[-1]["event"] == "failed"
    assert not any(record["event"] == "completed" for record in records)
    assert "private-session-value" not in output + json.dumps(records)
