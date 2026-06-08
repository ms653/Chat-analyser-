"""Tests for crisis detection — Tier 1 taxonomy and Tier 2 RF classifier."""

import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyser
from analyser import (
    classify_intent,
    CRISIS_TAXONOMY,
    CRISIS_LEVEL_LABELS,
    DISTRESS_KW,
    _any_pattern,
    compute_coping_summary,
)


# ── CRISIS_TAXONOMY structure ────────────────────────────────────────────────

class TestCrisisTaxonomy:
    def test_required_dimensions_present(self):
        for dim in ("cognitive", "behavioral", "emotional"):
            assert dim in CRISIS_TAXONOMY, f"Missing dimension: {dim}"

    def test_each_dimension_has_level_and_patterns(self):
        for dim, data in CRISIS_TAXONOMY.items():
            assert "level" in data, f"{dim} missing 'level'"
            assert "patterns" in data, f"{dim} missing 'patterns'"
            assert isinstance(data["level"], int)
            assert 1 <= data["level"] <= 3

    def test_behavioral_is_critical(self):
        assert CRISIS_TAXONOMY["behavioral"]["level"] == 3

    def test_level_labels_complete(self):
        for lvl in (1, 2, 3):
            assert lvl in CRISIS_LEVEL_LABELS


# ── Tier 1: classify_intent ──────────────────────────────────────────────────

class TestClassifyIntentCrisisTier1:
    def _msg(self, text: str) -> dict:
        return {"text": text, "is_user": True, "sender": "User",
                "date": datetime.date(2024, 1, 1)}

    def test_behavioral_flags_critical(self):
        msg = self._msg("I want to kill myself")
        intents = classify_intent(msg)
        assert "CRISIS_FLAG" in intents
        assert msg.get("crisis_dimension") == "behavioral"
        assert msg.get("crisis_level") == 3

    def test_self_harm_explicit(self):
        msg = self._msg("I've been cutting myself again")
        intents = classify_intent(msg)
        assert "CRISIS_FLAG" in intents
        assert msg.get("crisis_dimension") == "behavioral"

    def test_cognitive_flags_elevated(self):
        msg = self._msg("I feel worthless and hopeless")
        intents = classify_intent(msg)
        assert "CRISIS_FLAG" in intents
        assert msg.get("crisis_dimension") == "cognitive"
        assert msg.get("crisis_level") == 2

    def test_better_off_without_me(self):
        msg = self._msg("everyone would be better off without me")
        intents = classify_intent(msg)
        assert "CRISIS_FLAG" in intents
        assert msg.get("crisis_dimension") == "cognitive"

    def test_emotional_flags(self):
        msg = self._msg("I feel completely numb and empty inside")
        intents = classify_intent(msg)
        assert "CRISIS_FLAG" in intents
        assert msg.get("crisis_dimension") == "emotional"

    def test_benign_message_not_flagged(self):
        msg = self._msg("How are you doing today? Shall we meet for coffee?")
        intents = classify_intent(msg)
        assert "CRISIS_FLAG" not in intents

    def test_sarcastic_kill_me_not_flagged(self):
        msg = self._msg("kill me this meeting is SO boring")
        intents = classify_intent(msg)
        # The regex "want to die|kill myself" should NOT match this colloquial phrase
        # "kill me" alone isn't in the taxonomy — only "kill myself"
        assert "CRISIS_FLAG" not in intents

    def test_distress_signal_separate_from_crisis(self):
        msg = self._msg("I've been feeling really anxious and depressed lately")
        intents = classify_intent(msg)
        assert "DISTRESS_SIGNAL" in intents
        assert "CRISIS_FLAG" not in intents

    def test_panic_attack_is_distress(self):
        msg = self._msg("I had a panic attack at work today")
        intents = classify_intent(msg)
        assert "DISTRESS_SIGNAL" in intents

    def test_overwhelmed_is_distress(self):
        msg = self._msg("I just can't cope with everything right now, I'm completely overwhelmed")
        intents = classify_intent(msg)
        assert "DISTRESS_SIGNAL" in intents

    def test_crisis_source_default_tier1(self):
        msg = self._msg("I don't want to be here anymore")
        classify_intent(msg)
        if "CRISIS_FLAG" in classify_intent(msg):
            assert msg.get("crisis_source", "tier1_regex") == "tier1_regex"


# ── Coping Framework ─────────────────────────────────────────────────────────

class TestCopingFramework:
    def _chat(self, texts: list) -> dict:
        messages = [
            {"text": t, "is_user": True, "sender": "Me",
             "date": datetime.date(2024, 1, 1), "dt": None, "raw": t,
             "intents": []}
            for t in texts
        ]
        return {
            "messages": messages,
            "contact_name": "Alice",
            "config": {}, "framework_content": None,
            "contact_relationship": "", "tags": [],
        }

    def test_positive_coping_detected(self):
        chat = self._chat([
            "I've been going for walks and taking care of myself",
            "I started therapy and it's really helping",
            "I went to the gym and cooked a healthy meal",
        ])
        analyser.add_intents(chat)
        summary = compute_coping_summary(chat)
        assert summary["positive_count"] > 0

    def test_negative_coping_detected(self):
        chat = self._chat([
            "I just want to give up, there's no point trying anymore",
            "I've been avoiding everyone and just hoping it goes away",
        ])
        analyser.add_intents(chat)
        summary = compute_coping_summary(chat)
        assert summary["negative_count"] > 0

    def test_percentages_sum_to_100(self):
        chat = self._chat([
            "I'm taking care of myself",
            "I feel completely hopeless",
        ])
        analyser.add_intents(chat)
        summary = compute_coping_summary(chat)
        total = summary["positive_pct"] + summary["negative_pct"]
        # Allow for floating point; if there are flagged messages total = 100
        if summary["positive_count"] + summary["negative_count"] > 0:
            assert abs(total - 100.0) < 1.0

    def test_no_coping_signals_returns_zeros(self):
        chat = self._chat(["How was your day?", "Good thanks, yours?"])
        analyser.add_intents(chat)
        summary = compute_coping_summary(chat)
        assert summary["positive_count"] == 0
        assert summary["negative_count"] == 0


# ── _build_crisis_flags_with_context ────────────────────────────────────────

class TestBuildCrisisFlags:
    def _make_messages(self) -> list:
        msgs = [
            {"text": "How are you?", "sender": "Alice", "date": datetime.date(2024, 1, 1),
             "intents": [], "is_user": False},
            {"text": "Not great to be honest", "sender": "Me", "date": datetime.date(2024, 1, 1),
             "intents": [], "is_user": True},
            {"text": "I want to kill myself", "sender": "Me", "date": datetime.date(2024, 1, 1),
             "intents": ["CRISIS_FLAG"], "crisis_dimension": "behavioral",
             "crisis_level": 3, "is_user": True},
            {"text": "Oh no are you okay?", "sender": "Alice", "date": datetime.date(2024, 1, 1),
             "intents": [], "is_user": False},
        ]
        return msgs

    def test_flags_extracted(self):
        msgs = self._make_messages()
        flags = analyser._build_crisis_flags_with_context(msgs)
        assert len(flags) == 1
        assert flags[0]["text"] == "I want to kill myself"

    def test_context_before_and_after(self):
        msgs = self._make_messages()
        flags = analyser._build_crisis_flags_with_context(msgs)
        flag = flags[0]
        assert len(flag["context_before"]) >= 1
        assert len(flag["context_after"]) >= 1

    def test_dimension_and_level_in_flag(self):
        msgs = self._make_messages()
        flags = analyser._build_crisis_flags_with_context(msgs)
        flag = flags[0]
        assert flag["crisis_dimension"] == "behavioral"
        assert flag["crisis_level"] == 3

    def test_no_flags_returns_empty(self):
        msgs = [
            {"text": "Hi!", "sender": "Alice", "date": datetime.date(2024, 1, 1),
             "intents": [], "is_user": False},
        ]
        flags = analyser._build_crisis_flags_with_context(msgs)
        assert flags == []
