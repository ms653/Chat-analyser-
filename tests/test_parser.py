"""Tests for ProgrammaticWhatsAppParser and supporting parser utilities."""

import sys
import os
import datetime
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyser
from analyser import (
    _hformat_to_regex,
    _parse_dt,
    ProgrammaticWhatsAppParser,
    MSG_RE,
    _is_noise,
)


# ── _parse_dt ─────────────────────────────────────────────────────────────────

class TestParseDt:
    def test_24hr_with_seconds(self):
        dt = _parse_dt("25/12/2023", "14:30:05")
        assert dt == datetime.datetime(2023, 12, 25, 14, 30, 5)

    def test_24hr_no_seconds(self):
        dt = _parse_dt("01/06/24", "09:15")
        assert dt == datetime.datetime(2024, 6, 1, 9, 15)

    def test_12hr_am(self):
        dt = _parse_dt("01/01/2023", "08:30 AM")
        assert dt is not None
        assert dt.hour == 8

    def test_12hr_pm(self):
        dt = _parse_dt("01/01/2023", "02:30 PM")
        assert dt is not None
        assert dt.hour == 14

    def test_dot_separator(self):
        dt = _parse_dt("25.12.2023", "14:30:05")
        assert dt == datetime.datetime(2023, 12, 25, 14, 30, 5)

    def test_dash_separator(self):
        dt = _parse_dt("25-12-2023", "14:30:05")
        assert dt == datetime.datetime(2023, 12, 25, 14, 30, 5)

    def test_invalid_returns_none(self):
        assert _parse_dt("not-a-date", "not-a-time") is None


# ── MSG_RE global pattern ────────────────────────────────────────────────────

class TestMsgRe:
    def test_standard_format(self):
        line = "25/12/2023, 14:30:05 - Alice: Hello there"
        m = MSG_RE.match(line)
        assert m is not None
        assert m.group(3) == "Alice"
        assert m.group(4) == "Hello there"

    def test_bracket_format(self):
        line = "[25/12/23, 14:30:05] Bob: How are you?"
        m = MSG_RE.match(line)
        assert m is not None
        assert m.group(3) == "Bob"
        assert m.group(4) == "How are you?"

    def test_lrm_prefix_stripped(self):
        lrm = "‎"
        line = f"{lrm}25/12/2023, 14:30 - Carol: Hi"
        m = MSG_RE.match(line)
        assert m is not None
        assert m.group(3) == "Carol"

    def test_12hr_format(self):
        line = "25/12/2023, 2:30 PM - Dave: PM message"
        m = MSG_RE.match(line)
        assert m is not None

    def test_non_message_line_no_match(self):
        assert MSG_RE.match("This is just a continuation line") is None
        assert MSG_RE.match("") is None


# ── _is_noise ────────────────────────────────────────────────────────────────

class TestIsNoise:
    def test_media_omitted(self):
        assert _is_noise("<Media omitted>")
        assert _is_noise("image omitted")
        assert _is_noise("audio omitted")

    def test_system_messages(self):
        assert _is_noise("Messages and calls are end-to-end encrypted")
        assert _is_noise("missed voice call")
        assert _is_noise("missed video call")

    def test_normal_text_not_noise(self):
        assert not _is_noise("Hello, how are you?")
        assert not _is_noise("I'm doing well thanks")

    def test_lrm_line_is_noise(self):
        assert _is_noise("‎")


# ── _hformat_to_regex ────────────────────────────────────────────────────────

class TestHformatToRegex:
    def test_standard_ios_format(self):
        hformat = "[%d/%m/%y, %H:%M:%S] %name: %text"
        pattern = _hformat_to_regex(hformat)
        line = "[25/12/23, 14:30:05] Alice: Hello"
        m = pattern.match(line)
        assert m is not None
        assert m.group("sender") == "Alice"
        assert m.group("text") == "Hello"

    def test_android_format(self):
        hformat = "%d/%m/%Y, %H:%M - %name: %text"
        pattern = _hformat_to_regex(hformat)
        line = "25/12/2023, 14:30 - Bob: Hi there"
        m = pattern.match(line)
        assert m is not None
        assert m.group("sender") == "Bob"

    def test_12hr_format(self):
        hformat = "[%d/%m/%y, %I:%M:%S %p] %name: %text"
        pattern = _hformat_to_regex(hformat)
        line = "[25/12/23, 02:30:05 PM] Carol: Afternoon"
        m = pattern.match(line)
        assert m is not None
        assert m.group("sender") == "Carol"

    def test_non_matching_line(self):
        hformat = "[%d/%m/%y, %H:%M:%S] %name: %text"
        pattern = _hformat_to_regex(hformat)
        assert pattern.match("Just a regular line") is None

    def test_day_month_year_captured(self):
        hformat = "%d/%m/%Y, %H:%M - %name: %text"
        pattern = _hformat_to_regex(hformat)
        m = pattern.match("01/06/2024, 09:15 - Dave: Test")
        assert m.group("day") == "01"
        assert m.group("month") == "06"
        assert m.group("year") == "2024"


# ── ProgrammaticWhatsAppParser ───────────────────────────────────────────────

class TestProgrammaticWhatsAppParser:
    def _make_chat_file(self, lines: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write("\n".join(lines))
        f.close()
        return f.name

    def _make_config(self, path: str, **kwargs) -> dict:
        cfg = {
            "file": path,
            "contact_name": "Alice",
            "contact_relationship": "friend",
            "tags": [],
        }
        cfg.update(kwargs)
        return cfg

    def test_basic_parse(self):
        path = self._make_chat_file([
            "25/12/2023, 14:30:05 - Me: Hello Alice",
            "25/12/2023, 14:31:00 - Alice: Hey Me!",
        ])
        analyser.PRIMARY_USER_NAME = "Me"
        cfg = self._make_config(path)
        parser = ProgrammaticWhatsAppParser(cfg)
        result = parser.parse()
        assert len(result["messages"]) == 2
        assert result["messages"][0]["is_user"] is True
        assert result["messages"][1]["is_user"] is False
        os.unlink(path)

    def test_multiline_continuation(self):
        path = self._make_chat_file([
            "25/12/2023, 14:30:05 - Me: First line",
            "continuation of the message",
            "and another line",
            "25/12/2023, 14:31:00 - Alice: Reply",
        ])
        analyser.PRIMARY_USER_NAME = "Me"
        cfg = self._make_config(path)
        result = ProgrammaticWhatsAppParser(cfg).parse()
        assert len(result["messages"]) == 2
        assert "continuation of the message" in result["messages"][0]["text"]
        assert "and another line" in result["messages"][0]["text"]
        os.unlink(path)

    def test_ios_lrm_stripped_from_sender(self):
        lrm = "‎"
        path = self._make_chat_file([
            f"{lrm}25/12/2023, 14:30 - {lrm}Alice{lrm}: Hello",
        ])
        analyser.PRIMARY_USER_NAME = "Me"
        cfg = self._make_config(path)
        result = ProgrammaticWhatsAppParser(cfg).parse()
        if result["messages"]:
            assert "‎" not in result["messages"][0]["sender"]
        os.unlink(path)

    def test_noise_lines_excluded(self):
        path = self._make_chat_file([
            "25/12/2023, 14:30:05 - Me: Hello",
            "25/12/2023, 14:31:00 - Alice: image omitted",
            "25/12/2023, 14:32:00 - Me: Back again",
        ])
        analyser.PRIMARY_USER_NAME = "Me"
        cfg = self._make_config(path)
        result = ProgrammaticWhatsAppParser(cfg).parse()
        assert len(result["messages"]) == 2

    def test_hformat_custom(self):
        path = self._make_chat_file([
            "[25/12/23, 14:30:05] Me: Hi there",
            "[25/12/23, 14:31:00] Alice: Hello!",
        ])
        analyser.PRIMARY_USER_NAME = "Me"
        cfg = self._make_config(path, hformat="[%d/%m/%y, %H:%M:%S] %name: %text")
        result = ProgrammaticWhatsAppParser(cfg, cfg["hformat"]).parse()
        assert len(result["messages"]) == 2
        assert result["messages"][0]["sender"] == "Me"
        assert result["messages"][1]["sender"] == "Alice"
        os.unlink(path)

    def test_file_not_found_returns_empty(self):
        cfg = self._make_config("/nonexistent/path/chat.txt")
        result = ProgrammaticWhatsAppParser(cfg).parse()
        assert result["messages"] == []
        assert result["contact_name"] == "Alice"

    def test_parse_chat_function_delegates(self):
        path = self._make_chat_file([
            "25/12/2023, 14:30:05 - Me: Test",
        ])
        analyser.PRIMARY_USER_NAME = "Me"
        cfg = self._make_config(path)
        result = analyser.parse_chat(cfg)
        assert len(result["messages"]) == 1
        os.unlink(path)
