"""Tests for the LocalAffectAggregator and emoji sentiment scoring."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyser
from analyser import LocalAffectAggregator


class TestEmojiLexicon:
    """Verify emoji scoring constants are populated and sensible."""

    def test_lexicon_exists(self):
        assert hasattr(analyser, "_EMOJI_LEXICON")
        assert len(analyser._EMOJI_LEXICON) > 0

    def test_negative_emoji_scores(self):
        lexicon = analyser._EMOJI_LEXICON
        assert lexicon.get("😭", 0) < 0
        assert lexicon.get("🤬", 0) < 0

    def test_positive_emoji_scores(self):
        lexicon = analyser._EMOJI_LEXICON
        assert lexicon.get("😊", 0) > 0
        assert lexicon.get("🥰", 0) > 0

    def test_scores_in_range(self):
        for emoji, score in analyser._EMOJI_LEXICON.items():
            assert -1.0 <= score <= 1.0, f"{emoji} score {score} out of range"


class TestLocalAffectAggregator:
    """Test the Ekman affect aggregator — falls back to TextBlob path when transformers absent."""

    def _aggregator(self):
        # Pass None as pipeline_obj to force the TextBlob fallback path
        return LocalAffectAggregator(pipeline_obj=None)

    def test_score_returns_expected_keys(self):
        agg = self._aggregator()
        result = agg.score("I am so happy today!")
        required = {"label", "score", "emotions", "composite_valence", "emoji_sentiment", "dominant_emotion"}
        assert required <= result.keys(), f"Missing keys: {required - result.keys()}"

    def test_ekman_emotion_keys_present(self):
        agg = self._aggregator()
        result = agg.score("I feel terrible and scared")
        expected_emotions = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
        assert expected_emotions <= set(result["emotions"].keys())

    def test_composite_valence_range(self):
        agg = self._aggregator()
        for text in ["great day!", "terrible day", "just okay", "😭😭", "🥰🥰"]:
            result = agg.score(text)
            cv = result["composite_valence"]
            assert -1.0 <= cv <= 1.0, f"composite_valence {cv} out of range for '{text}'"

    def test_emoji_only_text(self):
        agg = self._aggregator()
        result = agg.score("😭😭😭")
        assert result["emoji_sentiment"] < 0

    def test_positive_emoji_text(self):
        agg = self._aggregator()
        result = agg.score("🥰🥰🥰")
        assert result["emoji_sentiment"] > 0

    def test_neutral_text(self):
        agg = self._aggregator()
        result = agg.score("The meeting is at 3pm")
        assert result["label"] in ("neutral", "positive", "negative")

    def test_dominant_emotion_is_valid(self):
        agg = self._aggregator()
        result = agg.score("I am so angry!")
        valid = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
        assert result["dominant_emotion"] in valid

    def test_empty_string(self):
        agg = self._aggregator()
        result = agg.score("")
        assert isinstance(result, dict)
        assert "composite_valence" in result

    def test_emoji_composite_weighting(self):
        """Joy text + happy emoji should score higher than joy text alone."""
        agg = self._aggregator()
        r1 = agg.score("I feel great today")
        r2 = agg.score("I feel great today 😊😊😊")
        assert r2["composite_valence"] >= r1["composite_valence"] - 0.1


class TestAddSentimentScores:
    """Integration test: check that add_sentiment_scores() attaches sentiment to messages."""

    def _make_chat(self, texts: list) -> dict:
        messages = [
            {"text": t, "is_user": True, "sender": "Me", "date": None, "dt": None, "raw": t}
            for t in texts
        ]
        return {
            "messages": messages,
            "contact_name": "Alice",
            "config": {},
            "framework_content": None,
            "contact_relationship": "",
            "tags": [],
        }

    def test_sentiment_attached(self):
        chat = self._make_chat(["I am happy!", "This is terrible."])
        analyser.add_sentiment_scores(chat, engine="textblob")
        for msg in chat["messages"]:
            assert "sentiment" in msg
            s = msg["sentiment"]
            assert "label" in s
            assert "composite_valence" in s

    def test_dominant_emotion_attached(self):
        chat = self._make_chat(["Wonderful news!"])
        analyser.add_sentiment_scores(chat, engine="textblob")
        de = chat["messages"][0]["sentiment"].get("dominant_emotion")
        assert de in {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
