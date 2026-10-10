"""Unit test for the Windows console-encoding safety helper."""

from __future__ import annotations

from console_utf8 import ensure_utf8_stdout


class _FakeStream:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.reconfigured_with = None

    def reconfigure(self, encoding=None, errors=None):
        self.reconfigured_with = (encoding, errors)
        self.encoding = encoding


def test_ensure_utf8_stdout_reconfigures_non_utf8_streams(monkeypatch):
    fake_out = _FakeStream("cp1252")
    fake_err = _FakeStream("cp1252")
    monkeypatch.setattr("sys.stdout", fake_out)
    monkeypatch.setattr("sys.stderr", fake_err)

    ensure_utf8_stdout()

    assert fake_out.reconfigured_with == ("utf-8", "replace")
    assert fake_err.reconfigured_with == ("utf-8", "replace")


def test_ensure_utf8_stdout_skips_already_utf8_streams(monkeypatch):
    fake_out = _FakeStream("utf-8")
    monkeypatch.setattr("sys.stdout", fake_out)

    ensure_utf8_stdout()

    assert fake_out.reconfigured_with is None


def test_ensure_utf8_stdout_does_not_raise_without_reconfigure(monkeypatch):
    class _NoReconfigure:
        encoding = "cp1252"

    monkeypatch.setattr("sys.stdout", _NoReconfigure())
    ensure_utf8_stdout()  # must not raise even though reconfigure() is absent
