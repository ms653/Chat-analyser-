#!/usr/bin/env python3
"""
WhatsApp Chat Analyser
Processes WhatsApp export files and generates a self-contained HTML analysis document.
Uses Gemma 4 via Ollama for local AI insight, with optional Claude API fallback for crisis assessment.
"""

import argparse
import json
import re
import sys
import os
import time
import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these values before running
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_USER_NAME = "Morgan Strutton"

CHATS = [
    {
        "file": "path/to/whatsapp_mum.txt",
        "contact_name": "Michelle O'Neill",
        "contact_relationship": "Mother",
        "tags": ["family", "primary"],
        "framework": "path/to/mum_framework.md",
    },
    {
        "file": "path/to/whatsapp_miranda.txt",
        "contact_name": "Miranda",
        "contact_relationship": "Sister",
        "tags": ["family"],
        "framework": None,
    },
]

OUTPUT_FILE = "analysis.html"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4"          # adjust to your installed model name
ANTHROPIC_API_KEY = ""           # optional — only for crisis assessment
CRISIS_AI_ASSESSMENT = False     # set True to enable Claude API for crisis review
NLP_ENGINE = "transformers"      # "transformers" or "textblob"
CUSTOM_TOPICS = []               # e.g. ["football", "therapy", "job hunting"]

# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  [DD/MM/YYYY, HH:MM:SS] Sender: Message
# Also handles two-digit year and 12-hr timestamps WhatsApp sometimes uses
MSG_RE = re.compile(
    r"^[‎‏]?\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?(?:\s[-–]\s|\s)"
    r"(.+?):\s(.+)$"
)


def _hformat_to_regex(hformat: str) -> re.Pattern:
    """Translate a simplified WhatsApp hformat string to a compiled regex.

    Tokens: %d %m %y %Y %H %I %M %S %p %name %text
    Captures named groups: date_str, time_str, sender, text.
    Example: "[%d/%m/%y, %H:%M:%S] %name: %text"
    """
    _TOKEN_MAP = {
        "%Y": r"(?P<year>\d{4})",
        "%y": r"(?P<year>\d{2,4})",
        "%d": r"(?P<day>\d{1,2})",
        "%m": r"(?P<month>\d{1,2})",
        "%H": r"(?P<hour>\d{1,2})",
        "%I": r"(?P<hour12>\d{1,2})",
        "%M": r"(?P<minute>\d{2})",
        "%S": r"(?P<second>\d{2})",
        "%p": r"(?P<ampm>[APap][Mm])",
        "%name": r"(?P<sender>.+?)",
        "%text": r"(?P<text>.+)",
    }
    parts = re.split(r"(%[a-zA-Z]+)", hformat)
    pattern = r"^[‎‏]?"
    for part in parts:
        pattern += _TOKEN_MAP.get(part, re.escape(part))
    pattern += r"$"
    return re.compile(pattern)


class ProgrammaticWhatsAppParser:
    """Class-based WhatsApp parser. Accepts optional hformat for custom date layouts.

    When hformat is None, falls back to the global MSG_RE auto-detection pattern.
    """

    def __init__(self, config: dict, hformat: Optional[str] = None):
        self.config = config
        self.hformat = hformat
        self._use_named = hformat is not None
        self.msg_re: re.Pattern = _hformat_to_regex(hformat) if hformat else MSG_RE

    def generate_regex_patterns(self, hformat: str) -> re.Pattern:
        return _hformat_to_regex(hformat)

    def _extract_groups(self, m: re.Match) -> Optional[tuple]:
        """Return (date_str, time_str, sender, text) from a match object."""
        try:
            if self._use_named:
                gd = m.groupdict()
                year = gd.get("year", "")
                day = gd.get("day", "")
                month = gd.get("month", "")
                hour = gd.get("hour") or gd.get("hour12") or "00"
                minute = gd.get("minute", "00")
                second = gd.get("second", "")
                ampm = gd.get("ampm", "")
                date_str = f"{day}/{month}/{year}" if day and month and year else ""
                time_parts = f"{hour}:{minute}"
                if second:
                    time_parts += f":{second}"
                if ampm:
                    time_parts += f" {ampm}"
                return date_str, time_parts, gd.get("sender", ""), gd.get("text", "")
            else:
                return m.group(1), m.group(2), m.group(3), m.group(4)
        except (IndexError, AttributeError):
            return None

    def parse(self) -> dict:
        """Parse the chat file and return a chat object."""
        path = self.config["file"]
        contact = self.config["contact_name"]
        messages: list = []

        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            print(f"[WARN] Chat file not found: {path}", file=sys.stderr)
            return _empty_chat(self.config)

        current = None
        for line in raw.splitlines():
            m = self.msg_re.match(line)
            if m:
                if current and not _is_noise(current["text"]):
                    messages.append(current)
                groups = self._extract_groups(m)
                if not groups:
                    continue
                date_str, time_str, sender, text = groups
                dt = _parse_dt(date_str, time_str)
                sender = sender.strip().strip('‎‏‪‬')
                is_user = sender.lower() == PRIMARY_USER_NAME.lower()
                current = {
                    "dt": dt,
                    "date": dt.date() if dt else None,
                    "sender": sender,
                    "is_user": is_user,
                    "text": text.strip(),
                    "raw": line,
                }
            elif current is not None:
                current["text"] += " " + line.strip()

        if current and not _is_noise(current["text"]):
            messages.append(current)

        framework_content = None
        fw_path = self.config.get("framework")
        if fw_path:
            try:
                framework_content = Path(fw_path).read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                print(f"[WARN] Framework file not found: {fw_path}", file=sys.stderr)

        return {
            "config": self.config,
            "messages": messages,
            "framework_content": framework_content,
            "contact_name": contact,
            "contact_relationship": self.config.get("contact_relationship", ""),
            "tags": self.config.get("tags", []),
        }

NOISE_PATTERNS = [
    re.compile(r"^<.+omitted>$", re.I),
    re.compile(r"^(image|video|audio|GIF|sticker|document|contact card) omitted$", re.I),
    re.compile(r"^(missed )?voice call$", re.I),
    re.compile(r"^(missed )?video call$", re.I),
    re.compile(r"^.+ changed the (subject|icon|description)", re.I),
    re.compile(r"^Messages and calls are end-to-end encrypted", re.I),
    re.compile(r"^Your security code with .+ changed", re.I),
    re.compile(r"^https?://\S+$"),
    re.compile(r"^null$", re.I),
    re.compile(r"^‎", re.I),  # left-to-right mark noise WhatsApp sometimes inserts
]


def _is_noise(text: str) -> bool:
    t = text.strip()
    for p in NOISE_PATTERNS:
        if p.match(t):
            return True
    return False


def _parse_dt(date_str: str, time_str: str) -> Optional[datetime.datetime]:
    """Try common WhatsApp date/time formats."""
    time_str = time_str.strip()
    date_str = date_str.strip()
    fmts = [
        "%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M:%S",
        "%d/%m/%Y %H:%M", "%d/%m/%y %H:%M",
        "%d/%m/%Y %I:%M:%S %p", "%d/%m/%y %I:%M:%S %p",
        "%d/%m/%Y %I:%M %p", "%d/%m/%y %I:%M %p",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S", "%d.%m.%y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%y %H:%M:%S",
    ]
    combined = f"{date_str} {time_str}"
    for fmt in fmts:
        try:
            return datetime.datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def detect_senders(file_path: str, sample_size: int = 400) -> list:
    """
    Quickly scan a chat file and return unique sender names, ordered by
    message count (most frequent first). Used by the GUI to auto-suggest
    names without a full parse.
    """
    try:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return []
    counts: dict = {}
    for line in raw.splitlines():
        m = MSG_RE.match(line)
        if m:
            sender = m.group(3).strip().strip('‎‏‪‬')
            text = m.group(4).strip()
            if not _is_noise(text):
                counts[sender] = counts.get(sender, 0) + 1
                if sum(counts.values()) >= sample_size:
                    break
    return [s for s, _ in sorted(counts.items(), key=lambda x: -x[1])]


def parse_chat(config: dict) -> dict:
    """Parse a single WhatsApp export file. Delegates to ProgrammaticWhatsAppParser."""
    return ProgrammaticWhatsAppParser(config, config.get("hformat")).parse()


def _empty_chat(config: dict) -> dict:
    return {
        "config": config,
        "messages": [],
        "framework_content": None,
        "contact_name": config.get("contact_name", "Unknown"),
        "contact_relationship": config.get("contact_relationship", ""),
        "tags": config.get("tags", []),
    }

# ─────────────────────────────────────────────────────────────────────────────
# NLP — SENTIMENT SCORING (Ekman 7-emotion + emoji composite valence)
# ─────────────────────────────────────────────────────────────────────────────

_EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_EMOJI_LEXICON = {
    "😭": -0.85, "😢": -0.80, "😤": -0.70, "🤬": -0.90, "🤢": -0.70,
    "😨": -0.65, "😰": -0.60, "😞": -0.55, "😔": -0.50, "😒": -0.40,
    "🙁": -0.35, "😕": -0.25, "😬": -0.20,
    "🥰": 0.90,  "😍": 0.85, "😊": 0.80, "😄": 0.78, "😀": 0.75,
    "😁": 0.72,  "🤗": 0.70, "👍": 0.65, "❤️": 0.88, "💕": 0.85,
    "💪": 0.60,  "🎉": 0.75, "🥳": 0.80, "😂": 0.50, "🤣": 0.55,
    "😮": 0.10,  "😲": 0.15, "😐": 0.00, "🤔": 0.05,
}


def _neutral_sentiment() -> dict:
    return {
        "label": "neutral", "score": 0.0,
        "emotions": {e: 0.0 for e in _EKMAN_LABELS},
        "composite_valence": 0.0,
        "emoji_sentiment": 0.0,
        "dominant_emotion": "neutral",
    }


class LocalAffectAggregator:
    """
    Ekman 7-emotion classifier with emoji-aware composite valence.
    Uses j-hartmann/emotion-english-distilroberta-base locally.
    Returns a sentiment dict that is backward-compatible (label + score fields preserved).
    """

    def __init__(self, pipeline_obj):
        self._pipeline = pipeline_obj  # HuggingFace pipeline or None (textblob fallback)

    def score(self, text: str) -> dict:
        if not text or len(text.strip()) < 3:
            return _neutral_sentiment()

        # Strip emojis and score them separately
        emoji_scores: list = []
        clean_text = text
        try:
            import emoji as emoji_lib
            clean_text = emoji_lib.replace_emoji(text, replace=" ").strip()
            for ch in text:
                if emoji_lib.is_emoji(ch):
                    emoji_scores.append(_EMOJI_LEXICON.get(ch, 0.0))
        except ImportError:
            pass

        # Emotion classification
        emotions = {e: 0.0 for e in _EKMAN_LABELS}
        if self._pipeline and len(clean_text.strip()) >= 3:
            try:
                raw = self._pipeline(clean_text[:512])
                # return_all_scores=True returns [[{label, score}, ...]]
                items = raw[0] if isinstance(raw[0], list) else raw
                for item in items:
                    lbl = item["label"].lower()
                    if lbl in emotions:
                        emotions[lbl] = round(item["score"], 4)
            except Exception:
                emotions["neutral"] = 1.0
        elif not self._pipeline:
            emotions["neutral"] = 1.0

        dominant = max(emotions, key=emotions.get)
        avg_emoji = sum(emoji_scores) / len(emoji_scores) if emoji_scores else 0.0

        # Composite valence: joy raises it, sadness + anger lower it, emojis contribute 30 %
        composite = round(
            0.7 * (emotions["joy"] - emotions["sadness"] - 0.5 * emotions["anger"])
            + 0.3 * avg_emoji,
            4,
        )
        composite = max(-1.0, min(1.0, composite))

        if composite > 0.1:
            label = "positive"
        elif composite < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": composite,           # backward-compat alias for composite_valence
            "emotions": emotions,
            "composite_valence": composite,
            "emoji_sentiment": round(avg_emoji, 4),
            "dominant_emotion": dominant,
        }


_affect_aggregator: object = None


def _load_affect_aggregator() -> LocalAffectAggregator:
    global _affect_aggregator
    if _affect_aggregator is not None:
        return _affect_aggregator  # type: ignore[return-value]
    try:
        from transformers import pipeline as hf_pipeline
        print("[INFO] Loading emotion model (j-hartmann/emotion-english-distilroberta-base)…")
        pipe = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        print("[INFO] Emotion model loaded.")
        _affect_aggregator = LocalAffectAggregator(pipe)
    except Exception as e:
        print(f"[WARN] Could not load emotion model: {e}. Falling back to TextBlob.")
        _affect_aggregator = LocalAffectAggregator(None)
    return _affect_aggregator  # type: ignore[return-value]


def score_sentiment_textblob(text: str) -> dict:
    """TextBlob fallback — maps polarity to a compatible Ekman-shaped dict."""
    try:
        from textblob import TextBlob
        polarity = round(TextBlob(text).sentiment.polarity, 4)
        joy = max(0.0, polarity)
        sadness = max(0.0, -polarity)
        neutral_score = round(1.0 - abs(polarity), 4)
        if polarity > 0.05:
            label, dominant = "positive", "joy"
        elif polarity < -0.05:
            label, dominant = "negative", "sadness"
        else:
            label, dominant = "neutral", "neutral"
        return {
            "label": label,
            "score": polarity,
            "emotions": {
                "anger": 0.0, "disgust": 0.0, "fear": 0.0,
                "joy": round(joy, 4), "neutral": neutral_score,
                "sadness": round(sadness, 4), "surprise": 0.0,
            },
            "composite_valence": polarity,
            "emoji_sentiment": 0.0,
            "dominant_emotion": dominant,
        }
    except Exception:
        return _neutral_sentiment()


def add_sentiment_scores(chat: dict, engine: str = "transformers") -> None:
    """Attach Ekman emotion scores and composite valence to every message."""
    if engine == "transformers":
        aggregator = _load_affect_aggregator()
        for msg in chat["messages"]:
            msg["sentiment"] = aggregator.score(msg["text"])
    else:
        for msg in chat["messages"]:
            msg["sentiment"] = score_sentiment_textblob(msg["text"])

# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFICATION (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

VISIT_OFFER_KW = [
    r"\bcome (over|round|visit|see you\b)",
    r"\bvisit\b", r"\bpop (round|over|in)\b",
    r"\bshall we (meet|see each other)\b",
    r"\bwant to (see|meet)\b",
    r"\bplan(ning)? (to|a) (visit|trip|come)\b",
    r"\bwhen (can|shall|are) (I|we) (come|visit|see)\b",
    r"\bwould you like me to (come|visit)\b",
]

VISIT_ACCEPT_KW = [
    r"\b(yes|yeah|yep|sure|definitely|great|sounds good|love (that|it)|perfect)\b.*visit",
    r"\bcome (over|round|visit|see you)\b.*\b(yes|yeah|sure|great)\b",
    r"\blooking forward\b",
    r"\bcan't wait\b",
    r"\bplease (come|visit|do)\b",
]

VISIT_CANCEL_KW = [
    r"\b(can't|cannot|won't be able|not (going to|gonna)|have to cancel|need to (cancel|postpone|reschedule))\b",
    r"\bmaybe another time\b",
    r"\bnot (a good|the best|great) time\b",
    r"\bactually.{0,30}(can't|busy|ill|unwell|tired)\b",
    r"\bsorry.{0,20}(can't|busy)\b",
    r"\btoo (tired|ill|unwell|busy)\b",
]

POST_VISIT_KW = [
    r"\bso (good|nice|lovely|wonderful|great) (to see|seeing) (you|each other)\b",
    r"\bwas (so|really|such) (good|nice|lovely|great|wonderful)\b",
    r"\bthank you for (having|coming|visiting)\b",
    r"\bso glad (you came|I came|we saw)\b",
    r"\bgreat (day|time|visit|afternoon|evening|weekend)\b",
    r"\bjust (left|got home|got back|got on the train)\b",
]

DISTRESS_KW = [
    r"\b(so |very |really )?(lonely|alone|isolated|empty|invisible|pointless|hopeless|worthless|nobody (cares|loves|wants))\b",
    r"\bno[- ]one (cares|is there|checks in|bothers)\b",
    r"\bwhat['']s (the point|the use)\b",
    r"\bfeel(ing)? (abandoned|forgotten|unloved|unwanted|like a burden)\b",
    r"\bcan['']t (go on|cope|do this|bear it|take it)\b",
    r"\bnever (get|feel|have|see)\b.{0,30}\b(better|happy|anyone|loved)\b",
    r"\bsince (dad|he|she|they) (died|left|passed|went)\b",
    # Mental health terms — clearly distress but below crisis threshold
    r"\b(anxious|anxiety|panicking|panic attack)\b",
    r"\b(depressed|depression|low mood|feeling low)\b",
    r"\b(overwhelmed|struggling|can['']t cope|not coping)\b",
    r"\b(self.worth|self.esteem|hate myself|hate my life)\b",
    r"\b(no (energy|motivation)|can['']t get (up|out of bed)|exhausted and (sad|low|down))\b",
    r"\b(nobody understands?|nobody listens?|nobody (cares?|notices?))\b",
]

# Structured crisis taxonomy — 3 active dimensions with risk levels
# Level 1 = Moderate, Level 2 = Elevated, Level 3 = Critical
CRISIS_TAXONOMY = {
    "cognitive": {
        "level": 2,
        "patterns": [
            r"\b(hopeless|worthless|not worth living|like a burden)\b",
            r"\b(better off (without me|dead))\b",
            r"\b(no (point|reason|purpose) (in living|to live|to go on|to be here))\b",
            r"\b(everyone.{0,40}better off without me|world.{0,30}better without me)\b",
            r"\b(don['']t want to be here( any ?more)?|just want to (disappear|not exist|be gone))\b",
        ],
    },
    "behavioral": {
        "level": 3,
        "patterns": [
            r"\b(want to die|kill myself|suicide|suicidal|ending it)\b",
            r"\b(took.{0,20}(pills|tablets)|self.harm)\b",
            r"\b(cut(ting)? myself|been cutting|i['']ve cut|started cutting|cut(s)? (again|(my )?(arm|wrist|leg|skin)))\b",
            r"\b(harm(ing)? myself|hurting myself|been hurting myself|hurt(ing)? my (arm|wrist|leg|skin))\b",
            r"\b(end(ing)? (it all|my life|everything))\b",
            r"\b(think(ing)? (about|of) (suicide|killing myself|ending (it|my life)|not being here))\b",
            r"\b(plan(ning)? to (hurt|harm|kill) myself)\b",
            r"\b(don['']t want to (be here|live|exist|carry on))\b",
        ],
    },
    "emotional": {
        "level": 2,
        "patterns": [
            r"\b(uncontrollable (crying|rage|anger)|can['']t stop (crying|shaking))\b",
            r"\b(completely (numb|empty|hollow)|feel(ing)? (nothing|completely empty|dead inside))\b",
            r"\b(overwhelming (despair|emptiness|pain))\b",
        ],
    },
}

CRISIS_LEVEL_LABELS = {1: "Moderate Risk", 2: "Elevated Risk", 3: "Critical Risk"}

# Kept for distress signals (sub-crisis; does not trigger CRISIS_FLAG)
CRISIS_KW: list = []  # no longer used directly — patterns live in CRISIS_TAXONOMY

FINANCIAL_KW = [
    r"\b(borrow|lend|loan|send|transfer|advance|owe|pay you back|need.{0,15}money|short.{0,10}this month)\b",
    r"\b(can you help.{0,20}(financially|with money|pay|afford))\b",
    r"\b(bills|rent|electricity|gas|mortgage).{0,20}(can['']t|struggling|behind|overdue)\b",
]

# DRIVE coping framework patterns
COPING_PATTERNS: dict = {
    "positive": {
        "self_care": [
            r"\b(self.?care|looking after (my|your)self|taking care of (my|your)self)\b",
            r"\b(had a (bath|walk|rest|nap)|went for a (walk|run)|getting (some )?rest)\b",
            r"\b(setting (a |my )?(boundary|boundaries|limits)|healthy (routine|habit))\b",
            r"\b(eating (well|better|healthily)|drinking (more )?water|exercise)\b",
        ],
        "seeking_help": [
            r"\b(therapy|therapist|counsell(ing|or)|seeing someone|talking to someone)\b",
            r"\b(doctor['']?s? (appointment|appointment)|GP|mental health (worker|team|support))\b",
            r"\b(rang (the )?(helpline|samaritans|crisis line)|called for help|crisis team)\b",
            r"\b(support group|group therapy|CBT|DBT|mindfulness class)\b",
        ],
        "prayer_meditation": [
            r"\b(pray(ed|ing)?|prayer|meditat(e|ed|ing|ion)|mindful(ness)?)\b",
            r"\b(deep breath(s|ing)?|breathing exercise|grounding (exercise|technique))\b",
            r"\b(spiritual(ity)?|my faith|god (helped|got me through)|gratitude)\b",
        ],
        "adaptive_humor": [
            r"\b(laugh(ed|ing)? about it|joking|dark hum(our|or)|makes me (laugh|smile))\b",
            r"\b(at least.{0,30}(funny|laugh)|silver lining|could (always) be worse)\b",
            r"\b(keeping (my|a) sense of hum(our|or)|find(ing)? the funny side)\b",
        ],
    },
    "negative": {
        "hopeless": [
            r"\b(no point|what['']s the point|nothing (will|is going to) change)\b",
            r"\b(nothing ever changes|give up|given up|don['']t see (the point|any hope))\b",
            r"\b(nothing I can do|out of my (control|hands)|useless (trying|fighting))\b",
            r"\b(always (been|going to be) (like this|the same|this way))\b",
        ],
        "avoidance": [
            r"\b(not thinking about (it|that)|trying not to think|ignore (it|that|the problem))\b",
            r"\b(deal with it (later|another day|tomorrow)|put(ting)? it (off|aside))\b",
            r"\b(binge.{0,15}(watching|eating)|losing myself in|just (escape|distract))\b",
            r"\b(if only (things|it|everything) (were|was) different|I wish things were different)\b",
        ],
        "conspiracy_paranoia": [
            r"\b(they['']re (all|out to|trying to)|nobody tells me (anything|the truth))\b",
            r"\b(hidden (agenda|motive)|working against me|everyone([ ']s| is) against me)\b",
            r"\b(conspiracy|cover.up|they (don['']t|never) want (me|us) to know)\b",
        ],
    },
}


def _any_pattern(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def classify_coping(text: str) -> dict:
    """Return DRIVE coping classification for a single message via regex."""
    for sub_theme, patterns in COPING_PATTERNS["positive"].items():
        if _any_pattern(text, patterns):
            return {"strategy": "positive", "sub_theme": sub_theme}
    for sub_theme, patterns in COPING_PATTERNS["negative"].items():
        if _any_pattern(text, patterns):
            return {"strategy": "negative", "sub_theme": sub_theme}
    return {"strategy": "none", "sub_theme": "none"}


def classify_intent(msg: dict) -> list:
    """Return a list of intent tags for a message; sets crisis metadata on the message dict."""
    intents = []
    t = msg["text"]
    if _any_pattern(t, VISIT_OFFER_KW):
        intents.append("VISIT_OFFERED")
    if _any_pattern(t, VISIT_ACCEPT_KW):
        intents.append("VISIT_ACCEPTED")
    if _any_pattern(t, VISIT_CANCEL_KW):
        intents.append("VISIT_CANCELLED")
    if _any_pattern(t, POST_VISIT_KW):
        intents.append("VISIT_HAPPENED")
    if _any_pattern(t, DISTRESS_KW):
        intents.append("DISTRESS_SIGNAL")
    # Tier 1: structured taxonomy — highest-level match wins
    for dimension, dim_data in CRISIS_TAXONOMY.items():
        if _any_pattern(t, dim_data["patterns"]):
            intents.append("CRISIS_FLAG")
            msg["crisis_dimension"] = dimension
            msg["crisis_level"] = dim_data["level"]
            break
    if _any_pattern(t, FINANCIAL_KW):
        intents.append("FINANCIAL_REQUEST")
    return intents


def add_intents(chat: dict) -> None:
    for msg in chat["messages"]:
        msg["intents"] = classify_intent(msg)

# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE DETECTION (>4hr gap = new exchange)
# ─────────────────────────────────────────────────────────────────────────────

EXCHANGE_GAP = datetime.timedelta(hours=4)


def label_exchanges(chat: dict) -> None:
    """Assign exchange_id and initiated_by to each message."""
    msgs = chat["messages"]
    if not msgs:
        return
    exchange_id = 0
    last_dt = None
    for msg in msgs:
        if last_dt is None or (msg["dt"] and (msg["dt"] - last_dt) > EXCHANGE_GAP):
            exchange_id += 1
            msg["exchange_id"] = exchange_id
            msg["exchange_start"] = True
            if msg["is_user"]:
                msg["intents"] = msg.get("intents", []) + ["INITIATED_BY_USER"]
            else:
                msg["intents"] = msg.get("intents", []) + ["INITIATED_BY_CONTACT"]
        else:
            msg["exchange_id"] = exchange_id
            msg["exchange_start"] = False
        if msg["dt"]:
            last_dt = msg["dt"]

# ─────────────────────────────────────────────────────────────────────────────
# PER-CHAT ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_visit_tracker(chat: dict) -> dict:
    """Chain VISIT_OFFERED → VISIT_ACCEPTED/CANCELLED/HAPPENED."""
    offered, accepted, cancelled, happened = 0, 0, 0, 0
    cancellations_by = Counter()
    events = []

    pending_offer = None
    for msg in chat["messages"]:
        intents = msg.get("intents", [])
        if "VISIT_OFFERED" in intents:
            pending_offer = msg
            offered += 1
            events.append({"date": str(msg["date"]), "type": "offered", "sender": msg["sender"]})
        if "VISIT_ACCEPTED" in intents and pending_offer:
            accepted += 1
            events.append({"date": str(msg["date"]), "type": "accepted", "sender": msg["sender"]})
            pending_offer = None
        if "VISIT_CANCELLED" in intents and pending_offer:
            cancelled += 1
            cancellations_by[msg["sender"]] += 1
            events.append({"date": str(msg["date"]), "type": "cancelled", "sender": msg["sender"]})
            pending_offer = None
        if "VISIT_HAPPENED" in intents:
            happened += 1
            events.append({"date": str(msg["date"]), "type": "happened", "sender": msg["sender"]})

    return {
        "offered": offered,
        "accepted": accepted,
        "cancelled": cancelled,
        "happened": happened,
        "cancellations_by": dict(cancellations_by),
        "events": events,
    }


def compute_initiation_balance(chat: dict) -> dict:
    user_starts = sum(1 for m in chat["messages"] if "INITIATED_BY_USER" in m.get("intents", []))
    contact_starts = sum(1 for m in chat["messages"] if "INITIATED_BY_CONTACT" in m.get("intents", []))
    total = user_starts + contact_starts or 1
    # Trend: compare first half vs second half of exchanges
    exchanges = [m for m in chat["messages"] if m.get("exchange_start")]
    half = len(exchanges) // 2
    if half:
        first_half_user = sum(1 for m in exchanges[:half] if "INITIATED_BY_USER" in m.get("intents", []))
        second_half_user = sum(1 for m in exchanges[half:] if "INITIATED_BY_USER" in m.get("intents", []))
        trend = "increasing" if second_half_user > first_half_user else (
            "decreasing" if second_half_user < first_half_user else "stable")
    else:
        trend = "stable"
    return {
        "user_pct": round(user_starts / total * 100, 1),
        "contact_pct": round(contact_starts / total * 100, 1),
        "user_count": user_starts,
        "contact_count": contact_starts,
        "trend": trend,
    }


def compute_response_times(chat: dict) -> dict:
    """Average response time per sender, plus list of >24hr gaps."""
    msgs = [m for m in chat["messages"] if m["dt"]]
    user_gaps, contact_gaps, long_gaps = [], [], []
    for i in range(1, len(msgs)):
        prev, curr = msgs[i - 1], msgs[i]
        if prev["exchange_id"] != curr.get("exchange_id"):
            continue  # different exchange — not a response
        gap = (curr["dt"] - prev["dt"]).total_seconds() / 3600  # hours
        if gap < 0 or gap > 72:
            continue
        if curr["is_user"]:
            user_gaps.append(gap)
        else:
            contact_gaps.append(gap)
        if gap > 24:
            long_gaps.append({
                "date": str(curr["date"]),
                "sender": curr["sender"],
                "hours": round(gap, 1),
            })
    return {
        "user_avg_hours": round(sum(user_gaps) / len(user_gaps), 1) if user_gaps else None,
        "contact_avg_hours": round(sum(contact_gaps) / len(contact_gaps), 1) if contact_gaps else None,
        "long_gaps": long_gaps,
    }


def compute_daily_sentiment(chat: dict) -> dict:
    """Aggregate sentiment scores per day per sender."""
    by_day_user = defaultdict(list)
    by_day_contact = defaultdict(list)
    for msg in chat["messages"]:
        if msg["date"] and "sentiment" in msg:
            s = msg["sentiment"]["score"]
            if msg["is_user"]:
                by_day_user[str(msg["date"])].append(s)
            else:
                by_day_contact[str(msg["date"])].append(s)
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
    user_series = {d: avg(v) for d, v in sorted(by_day_user.items())}
    contact_series = {d: avg(v) for d, v in sorted(by_day_contact.items())}
    return {"user": user_series, "contact": contact_series}


def compute_emotion_summary(chat: dict) -> dict:
    """Average Ekman emotion distribution (%) per sender across all messages."""
    user_totals = defaultdict(float)
    contact_totals = defaultdict(float)
    user_count = 0
    contact_count = 0
    for msg in chat["messages"]:
        emo = msg.get("sentiment", {}).get("emotions")
        if not emo:
            continue
        if msg["is_user"]:
            for e in _EKMAN_LABELS:
                user_totals[e] += emo.get(e, 0.0)
            user_count += 1
        else:
            for e in _EKMAN_LABELS:
                contact_totals[e] += emo.get(e, 0.0)
            contact_count += 1

    def avg_pct(totals, count):
        if not count:
            return {e: 0.0 for e in _EKMAN_LABELS}
        return {e: round(totals[e] / count * 100, 1) for e in _EKMAN_LABELS}

    return {
        "user": avg_pct(user_totals, user_count),
        "contact": avg_pct(contact_totals, contact_count),
    }


def compute_emotion_monthly(chat: dict) -> dict:
    """Monthly dominant-emotion counts, ready for stacked chart rendering."""
    month_counts: dict = defaultdict(Counter)
    for msg in chat["messages"]:
        dominant = msg.get("sentiment", {}).get("dominant_emotion", "neutral")
        date = msg.get("date")
        if not date:
            continue
        if isinstance(date, str):
            try:
                date = datetime.date.fromisoformat(date)
            except ValueError:
                continue
        month_counts[f"{date.year}-{date.month:02d}"][dominant] += 1
    months = sorted(month_counts.keys())
    series = {e: [month_counts[m].get(e, 0) for m in months] for e in _EKMAN_LABELS}
    return {"months": months, "series": series}


def compute_coping_summary(chat: dict) -> dict:
    """Run DRIVE coping classifier over every message and aggregate results."""
    pos_count = neg_count = 0
    by_subtheme: Counter = Counter()
    examples: dict = defaultdict(list)

    for msg in chat["messages"]:
        result = classify_coping(msg["text"])
        msg["coping"] = result
        strat = result["strategy"]
        sub = result["sub_theme"]
        if strat == "positive":
            pos_count += 1
            by_subtheme[sub] += 1
            if len(examples[sub]) < 3:
                examples[sub].append({"text": msg["text"][:200], "sender": msg["sender"]})
        elif strat == "negative":
            neg_count += 1
            by_subtheme[sub] += 1
            if len(examples[sub]) < 3:
                examples[sub].append({"text": msg["text"][:200], "sender": msg["sender"]})

    flagged = pos_count + neg_count or 1
    return {
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_pct": round(pos_count / flagged * 100, 1),
        "negative_pct": round(neg_count / flagged * 100, 1),
        "by_subtheme": dict(by_subtheme),
        "examples": dict(examples),
    }


def compute_weekly_volume(chat: dict) -> dict:
    """Message count per ISO week per sender."""
    user_by_week: dict = defaultdict(int)
    contact_by_week: dict = defaultdict(int)
    for msg in chat["messages"]:
        d = msg["date"]
        if not d:
            continue
        if isinstance(d, str):
            try:
                d = datetime.date.fromisoformat(d)
            except ValueError:
                continue
        iso = d.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        if msg["is_user"]:
            user_by_week[week_key] += 1
        else:
            contact_by_week[week_key] += 1
    all_weeks = sorted(set(list(user_by_week.keys()) + list(contact_by_week.keys())))
    return {
        "weeks": all_weeks,
        "user": [user_by_week.get(w, 0) for w in all_weeks],
        "contact": [contact_by_week.get(w, 0) for w in all_weeks],
    }


TOPIC_KEYWORDS = {
    "family_conflict": ["argument", "fight", "angry", "upset", "shouting", "screaming", "row", "fallout", "falling out"],
    "housing": ["house", "flat", "rent", "mortgage", "move", "landlord", "neighbour", "garden", "heating"],
    "mental_health": ["anxiety", "depressed", "depression", "therapy", "counselling", "panic", "overwhelmed", "struggling"],
    "kids": ["kids", "children", "school", "nursery", "grandkids", "grandson", "granddaughter"],
    "financial": ["money", "bills", "debt", "afford", "struggling", "overdraft", "benefits", "payment"],
    "positive_social": ["lovely", "wonderful", "great time", "good news", "exciting", "brilliant", "so happy", "so good"],
}


def compute_topics(chat: dict, custom_topics: list) -> dict:
    """Keyword-based topic frequency per message."""
    all_topics = dict(TOPIC_KEYWORDS)
    for t in custom_topics:
        all_topics[t.lower().replace(" ", "_")] = [t.lower()]
    topic_counts = Counter()
    topic_examples = defaultdict(list)
    for msg in chat["messages"]:
        t = msg["text"].lower()
        for topic, keywords in all_topics.items():
            if any(kw in t for kw in keywords):
                topic_counts[topic] += 1
                if len(topic_examples[topic]) < 3:
                    topic_examples[topic].append(msg["text"][:120])
    return {"counts": dict(topic_counts), "examples": dict(topic_examples)}


def run_lda_topics(messages: list, n_topics: int = 5) -> dict:
    """Probabilistic LDA topic discovery with monthly time-series.
    Returns {} gracefully if gensim or nltk are not installed or data is sparse.
    """
    try:
        from gensim import corpora, models as gensim_models
    except ImportError:
        return {}

    try:
        import nltk
        try:
            stop = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop = set(nltk.corpus.stopwords.words("english"))
    except ImportError:
        stop = set()

    _SKIP_PREFIXES = ("<Media omitted", "This message was deleted", "image omitted",
                      "audio omitted", "video omitted", "sticker omitted", "GIF omitted")

    # Words that are extremely common in casual chat but carry no topical meaning
    CHAT_STOP = {
        # high-frequency chat verbs
        "think", "know", "going", "want", "need", "like", "get", "got", "getting",
        "say", "said", "see", "saw", "go", "went", "come", "came", "coming",
        "make", "made", "take", "took", "give", "gave", "use", "used", "using",
        "try", "tried", "let", "feel", "felt", "look", "looked", "find", "found",
        "keep", "kept", "tell", "told", "ask", "asked", "thought", "put", "seem",
        "mean", "means", "meant", "hope", "loves", "love", "miss", "missed",
        # filler adjectives/adverbs
        "good", "great", "nice", "bad", "big", "little", "long", "old", "new",
        "sure", "right", "really", "actually", "literally", "basically", "probably",
        "just", "still", "even", "also", "bit", "lot", "way", "bit", "things", "thing",
        "today", "tonight", "tomorrow", "yesterday", "soon", "already", "always",
        "never", "maybe", "lol", "haha", "hahaha", "omg", "lmao", "xxx", "xox",
        # chat-specific noise
        "omitted", "media", "image", "video", "audio", "sticker", "gif",
        "message", "deleted", "null", "edited",
        # ultra-common pronouns/determiners not caught by NLTK
        "yeah", "yes", "okay", "well", "back", "home", "away", "here", "there",
        "something", "anything", "nothing", "everything", "someone", "anyone",
    }
    stop = stop | CHAT_STOP

    def _tokenize(text: str) -> list:
        tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())  # min length 4
        return [t for t in tokens if t not in stop]

    texts, msg_dates = [], []
    for m in messages:
        if any(m["text"].startswith(p) for p in _SKIP_PREFIXES):
            continue
        tokens = _tokenize(m["text"])
        if len(tokens) >= 3:
            texts.append(tokens)
            msg_dates.append(m["date"])

    if len(texts) < 20:
        return {}

    dictionary = corpora.Dictionary(texts)
    # Drop words in >60% of messages (too generic) or fewer than 3 messages (too rare)
    dictionary.filter_extremes(no_below=3, no_above=0.6)
    if len(dictionary) < 10:
        return {}
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = gensim_models.LdaModel(
        corpus, num_topics=n_topics, id2word=dictionary, passes=15, random_state=42
    )

    topic_labels = {
        i: " · ".join(w for w, _ in lda.show_topic(i, topn=4))
        for i in range(n_topics)
    }
    topic_words = {
        topic_labels[i]: [w for w, _ in lda.show_topic(i, topn=10)]
        for i in range(n_topics)
    }

    # Assign dominant topic per message
    dominant = []
    for bow in corpus:
        dist = dict(lda.get_document_topics(bow, minimum_probability=0))
        dominant.append(max(dist, key=dist.get) if dist else 0)

    # Monthly counts per topic
    month_counts: dict = defaultdict(lambda: Counter())
    for date, topic_idx in zip(msg_dates, dominant):
        if not date:
            continue
        if isinstance(date, str):
            try:
                date = datetime.date.fromisoformat(date)
            except ValueError:
                continue
        month_key = f"{date.year}-{date.month:02d}"
        month_counts[month_key][topic_idx] += 1

    months = sorted(month_counts.keys())
    series = {
        topic_labels[i]: [month_counts[m].get(i, 0) for m in months]
        for i in range(n_topics)
    }

    return {"months": months, "series": series, "topic_words": topic_words}


def extract_person_mentions(chat: dict, min_count: int = 4) -> dict:
    """
    Extract names mentioned ≥ min_count times.
    Heuristic: capitalised words that are not the known sender names, common words,
    or words that also appear lowercase in the corpus (i.e. sentence-start artefacts).
    """
    known = {PRIMARY_USER_NAME.lower(), chat["contact_name"].lower()}
    known.update({n.split()[0].lower() for n in known if n})

    STOP = {
        # pronouns / determiners
        "i", "i'm", "i've", "i'll", "i'd", "me", "my", "we", "our", "she", "he",
        "her", "his", "them", "they", "it", "that", "this", "what", "when",
        "where", "how", "who", "why",
        # discourse / affirmations
        "ok", "okay", "yes", "no", "hi", "hey", "oh", "so", "just", "got", "get",
        "don", "doesn", "didn", "isn", "wasn", "aren", "haven", "hadn", "wouldn",
        "couldn", "shouldn", "let", "going", "think", "know", "really", "well",
        "good", "great", "fine", "nice", "yeah", "yep", "nope", "right", "sure",
        # common emotional/social words that get capitalised mid-message
        "love", "dear", "thanks", "thank", "sorry", "hope", "happy", "sad",
        "god", "lord", "jesus", "christ", "bless",
        # seasonal / cultural nouns
        "xmas", "christmas", "easter", "halloween", "thanksgiving",
        # app/tech nouns
        "app", "phone", "message", "text", "chat", "call", "email", "whatsapp",
        # place-ish common nouns
        "home", "house", "road", "street", "town", "hospital", "doctor",
        # days / months
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }

    # Words that appear in lowercase anywhere in the corpus are ordinary words
    # that happen to be capitalised at sentence starts — not proper nouns.
    lowercase_corpus: set = set()
    for msg in chat["messages"]:
        for w in re.findall(r"\b[a-z]{3,}\b", msg["text"]):
            lowercase_corpus.add(w)

    name_counter = Counter()
    name_messages = defaultdict(list)

    for msg in chat["messages"]:
        words = re.findall(r"\b[A-Z][a-z]{2,}\b", msg["text"])
        for w in words:
            wl = w.lower()
            if wl not in known and wl not in STOP and wl not in lowercase_corpus:
                name_counter[w] += 1
                if len(name_messages[w]) < 8:
                    name_messages[w].append({
                        "text": msg["text"][:200],
                        "date": str(msg["date"]),
                        "sentiment": msg.get("sentiment", {}).get("label", "neutral"),
                    })

    people = {}
    for name, count in name_counter.most_common(30):
        if count >= min_count:
            msgs = name_messages[name]
            sentiments = [m["sentiment"] for m in msgs]
            sc = Counter(sentiments)
            total = len(sentiments) or 1
            people[name] = {
                "count": count,
                "sentiment_dist": {
                    "positive": round(sc.get("positive", 0) / total * 100, 1),
                    "neutral": round(sc.get("neutral", 0) / total * 100, 1),
                    "negative": round(sc.get("negative", 0) / total * 100, 1),
                },
                "excerpts": [m["text"] for m in msgs[:8]],
                "chats": [chat["contact_name"]],
            }
    return people


def run_absa(chat: dict, base_url: str, model: str) -> dict:
    """ABSA Call 8 — send message batches to Gemma; return entity→valence mapping."""
    messages = chat["messages"]
    BATCH = 15
    entity_data: dict = defaultdict(lambda: {"valences": [], "excerpts": []})

    for i in range(0, len(messages), BATCH):
        batch = messages[i:i + BATCH]
        formatted = "\n".join(
            f"[{j + 1}] {m['sender']}: {m['text'][:200]}"
            for j, m in enumerate(batch)
        )
        prompt = (
            "Analyse these chat messages and identify named people who are being TALKED ABOUT "
            "(not the message senders). For each mentioned person score how they are being "
            "discussed: valence from -1.0 (very negative) to +1.0 (very positive), 0 = neutral.\n"
            "Return ONLY valid JSON — an array, one object per (entity, message) pair found:\n"
            '[{"entity":"Name","valence":0.5,"reasoning":"brief reason","message_index":1}]\n'
            "If no named third-parties are discussed return: []\n\n"
            f"Messages:\n{formatted}"
        )
        try:
            raw = _ollama_chat(prompt, base_url, model)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end <= 0:
                continue
            results = json.loads(raw[start:end])
            for r in results:
                entity = str(r.get("entity", "")).strip()
                valence = r.get("valence")
                reasoning = str(r.get("reasoning", ""))
                msg_idx = int(r.get("message_index", 0)) - 1
                if not entity or valence is None:
                    continue
                try:
                    valence = max(-1.0, min(1.0, float(valence)))
                except (TypeError, ValueError):
                    continue
                entity_data[entity]["valences"].append(valence)
                if len(entity_data[entity]["excerpts"]) < 5 and 0 <= msg_idx < len(batch):
                    entity_data[entity]["excerpts"].append({
                        "text": batch[msg_idx]["text"][:200],
                        "reasoning": reasoning,
                        "valence": valence,
                    })
        except Exception as e:
            print(f"[WARN] ABSA batch {i // BATCH + 1} failed: {e}")

    result = {}
    for entity, data in entity_data.items():
        if not data["valences"]:
            continue
        avg = sum(data["valences"]) / len(data["valences"])
        result[entity] = {
            "avg_valence": round(avg, 3),
            "valence_history": data["valences"],
            "aspect_excerpts": data["excerpts"],
        }
    return result


def merge_absa_into_people(people: dict, absa_per_chat: list) -> None:
    """Merge per-chat ABSA valence results into the merged people dict (in-place)."""
    combined: dict = defaultdict(lambda: {"valences": [], "excerpts": []})
    for absa in absa_per_chat:
        for entity, data in absa.items():
            combined[entity]["valences"].extend(data.get("valence_history", []))
            combined[entity]["excerpts"].extend(data.get("aspect_excerpts", []))

    for entity, data in combined.items():
        if not data["valences"]:
            continue
        avg = sum(data["valences"]) / len(data["valences"])
        # Try exact match then case-insensitive
        matched = entity if entity in people else next(
            (k for k in people if k.lower() == entity.lower()), None
        )
        if matched:
            people[matched]["avg_valence"] = round(avg, 3)
            people[matched]["valence_history"] = data["valences"]
            people[matched]["aspect_excerpts"] = data["excerpts"][:8]


# ── Tier 2: Random Forest crisis classifier ──────────────────────────────────

_CRISIS_SEED: list = [
    # (text, label)  1=crisis  0=not-crisis
    ("I've been cutting myself again", 1),
    ("I want to die I really can't do this anymore", 1),
    ("everyone would be better off without me", 1),
    ("I took too many pills last night", 1),
    ("I don't want to be here anymore", 1),
    ("I've been hurting myself when things get bad", 1),
    ("I'm thinking about ending it all", 1),
    ("I have a plan to hurt myself tonight", 1),
    ("I just want to not exist anymore", 1),
    ("I've started cutting my arms again", 1),
    ("I don't see any point in living", 1),
    ("I've been self harming for weeks", 1),
    ("I made a decision to end things on Friday", 1),
    ("I'm not going to be around much longer", 1),
    ("I bought pills to take them all at once", 1),
    ("This homework is literally killing me lol", 0),
    ("I want to kill my boss sometimes I swear", 0),
    ("dying of laughter at this video", 0),
    ("kill me now this meeting is so boring", 0),
    ("I'm exhausted and feeling really low today", 0),
    ("feeling lonely and struggling a bit", 0),
    ("can't cope with all this stress at work", 0),
    ("I feel so empty and lost lately", 0),
    ("nobody seems to care about me right now", 0),
    ("that film was so sad it made me cry", 0),
    ("having such a terrible day", 0),
    ("this weather is so depressing", 0),
    ("ugh I feel completely dead inside today", 0),
    ("work is making me want to scream", 0),
    ("I'm exhausted and completely overwhelmed", 0),
    ("feeling a bit hopeless about the situation", 0),
    ("I've been really struggling lately with everything", 0),
    ("I just wish things were different", 0),
    ("just had a massive panic attack", 0),
    ("she hasn't called in weeks I'm so worried", 0),
    ("so angry I could scream", 0),
    ("this is absolutely killing my productivity", 0),
    ("I need to die laughing at this meme", 0),
    ("I feel like I'm drowning in work", 0),
]

_crisis_clf = None


def _clean_text_for_rf(text: str) -> str:
    """Preprocess text for TF-IDF: lowercase, strip noise, normalise repetition."""
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)
    t = re.sub(r"[^a-z\s''-]", " ", t)
    return t.strip()


def _load_crisis_classifier():
    """Load or train the Tier 2 Random Forest crisis classifier."""
    global _crisis_clf
    if _crisis_clf is not None:
        return _crisis_clf
    model_path = Path.home() / ".whatsapp_analyser_crisis_model.pkl"
    try:
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        if model_path.exists():
            with open(model_path, "rb") as f:
                _crisis_clf = pickle.load(f)
            return _crisis_clf

        # Train on seed data
        print("[INFO] Training Tier 2 crisis classifier…")
        texts = [_clean_text_for_rf(t) for t, _ in _CRISIS_SEED]
        labels = [l for _, l in _CRISIS_SEED]
        _crisis_clf = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=500)),
            ("clf", RandomForestClassifier(n_estimators=120, random_state=42)),
        ])
        _crisis_clf.fit(texts, labels)
        with open(model_path, "wb") as f:
            import pickle as pk
            pk.dump(_crisis_clf, f)
        print("[INFO] Tier 2 classifier trained and cached.")
    except ImportError:
        print("[WARN] scikit-learn not installed — Tier 2 crisis classifier disabled.")
        _crisis_clf = None
    except Exception as e:
        print(f"[WARN] Tier 2 classifier error: {e}")
        _crisis_clf = None
    return _crisis_clf


def add_tier2_crisis_scores(chat: dict) -> None:
    """Run Tier 2 RF on messages not already flagged by Tier 1; mark high-confidence positives."""
    clf = _load_crisis_classifier()
    if clf is None:
        return
    try:
        unflagged = [
            (i, msg) for i, msg in enumerate(chat["messages"])
            if "CRISIS_FLAG" not in msg.get("intents", [])
        ]
        if not unflagged:
            return
        texts = [_clean_text_for_rf(msg["text"]) for _, msg in unflagged]
        probs = clf.predict_proba(texts)
        crisis_idx = list(clf.classes_).index(1)
        for (i, msg), prob in zip(unflagged, probs):
            score = prob[crisis_idx]
            msg["crisis_rf_score"] = round(score, 3)
            if score >= 0.65:
                msg.setdefault("intents", []).append("CRISIS_FLAG")
                msg["crisis_dimension"] = "behavioral"
                msg["crisis_level"] = 3
                msg["crisis_source"] = "tier2_rf"
    except Exception as e:
        print(f"[WARN] Tier 2 scoring failed: {e}")


def _build_crisis_flags_with_context(messages: list) -> list:
    """Build crisis flag records including surrounding message context."""
    flags = []
    for i, m in enumerate(messages):
        if "CRISIS_FLAG" not in m.get("intents", []):
            continue
        flags.append({
            "date": str(m["date"]),
            "text": m["text"][:300],
            "sender": m["sender"],
            "crisis_dimension": m.get("crisis_dimension", "unknown"),
            "crisis_level": m.get("crisis_level", 2),
            "crisis_source": m.get("crisis_source", "tier1_regex"),
            "context_before": [
                {"sender": messages[j]["sender"], "text": messages[j]["text"][:200]}
                for j in range(max(0, i - 3), i)
            ],
            "context_after": [
                {"sender": messages[j]["sender"], "text": messages[j]["text"][:200]}
                for j in range(i + 1, min(len(messages), i + 3))
            ],
        })
    return flags


def run_per_chat_analysis(chat: dict, engine: str, custom_topics: list) -> None:
    """Run all per-chat Python analysis, attaching results to the chat dict."""
    print(f"[INFO] Analysing chat: {chat['contact_name']}")
    add_sentiment_scores(chat, engine)
    add_intents(chat)
    label_exchanges(chat)
    add_tier2_crisis_scores(chat)

    # Re-run intent after exchange labelling (exchange start intents added in label_exchanges)
    chat["analytics"] = {
        "visit_tracker": compute_visit_tracker(chat),
        "initiation_balance": compute_initiation_balance(chat),
        "response_times": compute_response_times(chat),
        "daily_sentiment": compute_daily_sentiment(chat),
        "weekly_volume": compute_weekly_volume(chat),
        "topics": compute_topics(chat, custom_topics),
        "lda_topics": run_lda_topics(chat["messages"]),
        "people": extract_person_mentions(chat),
        "emotion_summary": compute_emotion_summary(chat),
        "emotion_monthly": compute_emotion_monthly(chat),
        "coping_summary": compute_coping_summary(chat),
        "distress_signals": [
            {"date": str(m["date"]), "text": m["text"][:300], "sender": m["sender"]}
            for m in chat["messages"] if "DISTRESS_SIGNAL" in m.get("intents", [])
        ],
        "crisis_flags": _build_crisis_flags_with_context(chat["messages"]),
        "financial_requests": [
            {"date": str(m["date"]), "text": m["text"][:300], "sender": m["sender"]}
            for m in chat["messages"] if "FINANCIAL_REQUEST" in m.get("intents", [])
        ],
    }

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CHAT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def merge_people_across_chats(chats: list) -> dict:
    """Merge person cards across all chats."""
    merged = {}
    for chat in chats:
        for name, data in chat["analytics"]["people"].items():
            if name in merged:
                merged[name]["count"] += data["count"]
                merged[name]["chats"] = list(set(merged[name]["chats"] + data["chats"]))
                for s in ("positive", "neutral", "negative"):
                    merged[name]["sentiment_dist"][s] = round(
                        (merged[name]["sentiment_dist"][s] + data["sentiment_dist"][s]) / 2, 1
                    )
                merged[name]["excerpts"] = (merged[name]["excerpts"] + data["excerpts"])[:8]
            else:
                merged[name] = dict(data)
    return merged


def find_cross_chat_correlations(chats: list) -> list:
    """Look for dates where distress signals or initiation shifts appear across multiple chats."""
    if len(chats) < 2:
        return []
    # Build per-date distress map
    distress_by_date = defaultdict(list)
    for chat in chats:
        for d in chat["analytics"]["distress_signals"]:
            distress_by_date[d["date"]].append(chat["contact_name"])
    # Dates with distress in multiple chats
    correlations = []
    for date, contacts in distress_by_date.items():
        if len(contacts) > 1:
            correlations.append({
                "date": date,
                "type": "multi_chat_distress",
                "chats": contacts,
                "summary": f"Distress signals appear in multiple chats on {date}",
            })
    return correlations


def find_shared_people_across_chats(chats: list) -> list:
    """People appearing in multiple chats."""
    name_to_chats = defaultdict(list)
    for chat in chats:
        for name in chat["analytics"]["people"]:
            name_to_chats[name].append(chat["contact_name"])
    return [
        {"name": name, "chats": chats}
        for name, chats in name_to_chats.items()
        if len(chats) > 1
    ]


def run_cross_chat_analysis(chats: list) -> dict:
    print("[INFO] Running cross-chat analysis…")
    return {
        "merged_people": merge_people_across_chats(chats),
        "correlations": find_cross_chat_correlations(chats),
        "shared_people": find_shared_people_across_chats(chats),
    }

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA API CALLS
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_LOG = []


def _ollama_chat(messages: list, base_url: str, model: str, json_mode: bool = True) -> Optional[str]:
    """Single Ollama /api/chat call. Returns response text or None on failure."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        if "--log-tokens" in sys.argv:
            usage = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            TOKEN_LOG.append({"model": model, "tokens": usage})
        return content
    except requests.exceptions.ConnectionError:
        print(f"[WARN] Ollama not reachable at {base_url}. Skipping AI call.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[WARN] Ollama call failed: {e}", file=sys.stderr)
        return None


def _parse_json_response(raw: Optional[str], fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\}|\[[\s\S]+?\])\s*```", raw)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return fallback


def ai_people_cards(people: dict, base_url: str, model: str) -> dict:
    """Call 1 — enrich person cards with Gemma summaries."""
    results = {}
    for name, data in people.items():
        if not data["excerpts"]:
            continue
        prompt = (
            f"Based on these message excerpts and sentiment data, write a 2-3 sentence summary "
            f"of how the message author talks about {name} and what the relationship dynamic appears to be. "
            f"Suggest a one-phrase sentiment label. Flag any patterns worth watching. "
            f"Return as JSON: {{\"summary\": \"...\", \"sentiment_label\": \"...\", \"warning_flag\": \"...\"}}\n\n"
            f"Sentiment distribution: {data['sentiment_dist']}\n"
            f"Appears in chats: {', '.join(data['chats'])}\n"
            f"Excerpts:\n" + "\n".join(f"- {e}" for e in data["excerpts"][:6])
        )
        raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
        results[name] = _parse_json_response(raw, {"summary": "", "sentiment_label": "", "warning_flag": ""})
    return results


def ai_narrative_vs_record(chat: dict, cross_chat: dict, base_url: str, model: str) -> list:
    """Call 2 — narrative vs behavioural record contrasts."""
    visits = chat["analytics"]["visit_tracker"]
    init = chat["analytics"]["initiation_balance"]
    distress = chat["analytics"]["distress_signals"][:5]
    post_visit = [
        {"date": str(m["date"]), "text": m["text"][:200]}
        for m in chat["messages"] if "VISIT_HAPPENED" in m.get("intents", [])
    ][:3]
    # Cross-chat discrepancies relevant to this chat
    relevant_corr = [c for c in cross_chat.get("correlations", []) if chat["contact_name"] in c.get("chats", [])]
    prompt = (
        "Based on this behavioural data from a messaging conversation, identify up to 4 contrasts "
        "between what the pattern of behaviour suggests and what might be claimed verbally. "
        "Write each as a contrast pair. Be factual and measured. "
        "Return as JSON array: [{\"claimed\": \"...\", \"record_shows\": \"...\"}]\n\n"
        f"Visit summary: offered={visits['offered']}, accepted={visits['accepted']}, "
        f"cancelled={visits['cancelled']}, happened={visits['happened']}\n"
        f"Cancellations by: {visits['cancellations_by']}\n"
        f"Initiation: user {init['user_pct']}%, contact {init['contact_pct']}%, trend={init['trend']}\n"
        f"Distress signals:\n" + "\n".join(f"- [{d['date']}] {d['text'][:120]}" for d in distress) + "\n"
        f"Post-visit messages:\n" + "\n".join(f"- [{p['date']}] {p['text'][:120]}" for p in post_visit) + "\n"
        f"Cross-chat correlations: {json.dumps(relevant_corr)}"
    )
    raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
    return _parse_json_response(raw, [])


def ai_framework_personalisation(chat: dict, base_url: str, model: str) -> list:
    """Call 3 — framework personalisation suggestions."""
    if not chat.get("framework_content"):
        return []
    visits = chat["analytics"]["visit_tracker"]
    init = chat["analytics"]["initiation_balance"]
    rt = chat["analytics"]["response_times"]
    distress = chat["analytics"]["distress_signals"][:5]
    escalations = [
        {"date": str(m["date"]), "text": m["text"][:200]}
        for m in chat["messages"] if "CRISIS_FLAG" in m.get("intents", [])
    ][:3]
    prompt = (
        "Based on this behavioural data, suggest specific adjustments to the following communications framework. "
        "Keep suggestions brief and practical. Only suggest changes where the data clearly supports them. "
        "Return as JSON array: [{\"framework_item\": \"...\", \"suggested_adjustment\": \"...\", \"data_basis\": \"...\"}]\n\n"
        f"Initiation: user {init['user_pct']}%, contact {init['contact_pct']}%, trend={init['trend']}\n"
        f"Avg response gap: user={rt['user_avg_hours']}h, contact={rt['contact_avg_hours']}h\n"
        f"Visit outcome: offered={visits['offered']}, cancelled={visits['cancelled']}\n"
        f"Common distress phrases:\n" + "\n".join(f"- {d['text'][:100]}" for d in distress) + "\n"
        f"Escalation excerpts:\n" + "\n".join(f"- {e['text'][:100]}" for e in escalations) + "\n\n"
        f"Framework:\n{chat['framework_content'][:2000]}"
    )
    raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
    return _parse_json_response(raw, [])


def ai_timeline_highlights(chats: list, base_url: str, model: str) -> list:
    """Call 4 — significant timeline moments across all chats."""
    events = []
    for chat in chats:
        for msg in chat["messages"]:
            intents = msg.get("intents", [])
            relevant = [i for i in intents if i not in ("INITIATED_BY_USER", "INITIATED_BY_CONTACT")]
            if relevant:
                events.append({
                    "date": str(msg["date"]),
                    "chat": chat["contact_name"],
                    "intents": relevant,
                    "sentiment": msg.get("sentiment", {}).get("label", "neutral"),
                })
    # Deduplicate by date+chat
    seen = set()
    deduped = []
    for e in events:
        k = (e["date"], e["chat"], tuple(e["intents"]))
        if k not in seen:
            seen.add(k)
            deduped.append(e)
    deduped = deduped[:200]  # cap for token budget
    prompt = (
        "From this list of dates and classifications across one or more conversations, identify the 10-12 most "
        "significant moments — turning points, escalations, resolutions, or important patterns. "
        "Return as JSON array: [{\"date\": \"...\", \"chat\": \"...\", \"significance\": \"...\", "
        "\"type\": \"positive|warning|crisis|normal\"}]\n\n"
        f"Events:\n{json.dumps(deduped, indent=2)}"
    )
    raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
    return _parse_json_response(raw, [])


def ai_cross_chat_links(cross_chat: dict, chats: list, base_url: str, model: str) -> list:
    """Call 5 — cross-chat link analysis (only if >1 chat)."""
    if len(chats) < 2:
        return []
    correlations = cross_chat.get("correlations", [])
    shared = cross_chat.get("shared_people", [])
    if not correlations and not shared:
        return []
    prompt = (
        "Based on these dates where patterns appear to correlate across different conversations, "
        "identify which links are most significant and why. Describe each in one sentence. "
        "Return as JSON array: [{\"date_range\": \"...\", \"chats_involved\": \"...\", "
        "\"pattern\": \"...\", \"significance\": \"...\"}]\n\n"
        f"Correlations: {json.dumps(correlations)}\n"
        f"Shared people across chats: {json.dumps(shared)}"
    )
    raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
    return _parse_json_response(raw, [])


def _format_flag_for_llm(flag: dict, anonymise: bool = False) -> str:
    """Format a crisis flag with surrounding context for an LLM prompt."""
    sender_map: dict = {}

    def label(sender: str) -> str:
        if not anonymise:
            return sender
        if sender not in sender_map:
            sender_map[sender] = f"Person {chr(65 + len(sender_map))}"
        return sender_map[sender]

    lines = []
    for m in flag.get("context_before", []):
        lines.append(f"  [{label(m['sender'])}]: {m['text']}")
    lines.append(f">>> [{label(flag.get('sender', '?'))}]: {flag['text']} <<<")
    for m in flag.get("context_after", []):
        lines.append(f"  [{label(m['sender'])}]: {m['text']}")
    return "\n".join(lines)


_CRISIS_PROMPT_INSTRUCTIONS = (
    "You are reviewing messages flagged for potential crisis language. "
    "For each flagged message (marked >>>), read it alongside its surrounding context. "
    "Assess whether it represents genuine distress or risk, rhetorical/emotional expression "
    "(e.g. sarcasm, hyperbole), or is unclear. "
    "Consider: directness of language, repetition of similar themes across the excerpt, "
    "and whether context suggests escalation or past behaviour. "
    "Return ONLY a JSON array — no prose, no markdown — with one object per message:\n"
    "[{\"excerpt_id\": N, \"assessment\": \"1-2 sentence plain assessment\", "
    "\"confidence\": \"high|medium|low\", \"is_literal_risk\": true|false}]\n"
    "Do not reproduce harmful method information."
)


def ai_crisis_assessment_claude(crisis_flags: list, api_key: str) -> list:
    """Call 6 — Claude API crisis assessment with context (optional)."""
    if not crisis_flags or not api_key:
        return []
    try:
        import anthropic
    except ImportError:
        print("[WARN] anthropic package not installed. Skipping crisis assessment.", file=sys.stderr)
        return []
    client = anthropic.Anthropic(api_key=api_key)
    formatted = [
        {"id": i, "context": _format_flag_for_llm(f, anonymise=True)}
        for i, f in enumerate(crisis_flags)
    ]
    prompt = f"{_CRISIS_PROMPT_INSTRUCTIONS}\n\nFlagged messages:\n{json.dumps(formatted)}"
    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return _parse_json_response(raw, [])
    except Exception as e:
        print(f"[WARN] Claude API call failed: {e}", file=sys.stderr)
        return []


def ai_crisis_assessment_ollama(crisis_flags: list, base_url: str, model: str) -> list:
    """Local Gemma crisis assessment via Ollama — runs automatically when AI is enabled."""
    if not crisis_flags:
        return []
    results = []
    for i, flag in enumerate(crisis_flags):
        context_block = _format_flag_for_llm(flag, anonymise=False)
        prompt = (
            f"{_CRISIS_PROMPT_INSTRUCTIONS}\n\n"
            f"There is 1 flagged message (excerpt_id 0):\n{context_block}"
        )
        raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
        parsed = _parse_json_response(raw, None)
        # Ollama may return a single object or a list with one item
        if isinstance(parsed, list) and parsed:
            item = parsed[0]
        elif isinstance(parsed, dict):
            item = parsed
        else:
            item = {"assessment": "Could not assess.", "confidence": "low", "is_literal_risk": False}
        item["excerpt_id"] = i
        results.append(item)
    return results

def ai_coping_analysis(chat: dict, base_url: str, model: str) -> dict:
    """Call 7 — Gemma holistic DRIVE coping assessment for one chat."""
    summary = chat["analytics"].get("coping_summary", {})
    sample = []
    for msg in chat["messages"]:
        c = msg.get("coping", {})
        if c.get("strategy") != "none" and len(sample) < 12:
            sample.append(f"[{c['strategy'].upper()}/{c['sub_theme']}] {msg['text'][:150]}")
    if not sample:
        return {}
    prompt = (
        "Analyse the following chat messages classified by DRIVE coping strategy "
        "(Demands-Resources-Individual Effects model). Provide a holistic assessment.\n\n"
        f"Positive coping signals: {summary.get('positive_count', 0)} messages\n"
        f"Negative coping signals: {summary.get('negative_count', 0)} messages\n"
        f"Sub-themes: {list(summary.get('by_subtheme', {}).keys())}\n\n"
        "Sample classified messages:\n" + "\n".join(sample) + "\n\n"
        "Return JSON only: {\"dominant_strategy\": \"Positive|Negative|Mixed|None\", "
        "\"primary_sub_theme\": \"...\", "
        "\"assessment\": \"2-3 sentence plain-language summary\", "
        "\"recommendations\": \"1-2 sentence supportive suggestion\"}"
    )
    raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model)
    return _parse_json_response(raw, {})


def ai_relationship_summary(chat: dict, base_url: str, model: str) -> str:
    """Call 9 — Write a 2-3 paragraph relationship summary for a single chat."""
    msgs = chat["messages"]
    contact = chat["contact_name"]
    relationship = chat.get("contact_relationship", "")

    # Date range
    dates = [m["date"] for m in msgs if m["date"]]
    date_from = str(min(dates)) if dates else "unknown"
    date_to   = str(max(dates)) if dates else "unknown"

    # Topic keywords
    topics = list((chat["analytics"].get("topics") or {}).get("counts", {}).keys())[:8]

    # Sentiment stats
    a = chat["analytics"]
    init = a.get("initiation_balance", {})
    ds_count = len(a.get("distress_signals", []))
    cf_count = len(a.get("crisis_flags", []))

    # Sample messages spread across the timeline — early, middle, recent + emotionally significant
    n = len(msgs)
    indices = set()
    # Early 4, middle 4, recent 4
    for i in range(min(4, n)):
        indices.add(i)
    for i in range(max(0, n // 2 - 2), min(n, n // 2 + 2)):
        indices.add(i)
    for i in range(max(0, n - 4), n):
        indices.add(i)
    # Emotionally significant: high abs sentiment
    scored = sorted(
        range(n),
        key=lambda i: abs(msgs[i].get("sentiment", {}).get("score", 0) or 0),
        reverse=True,
    )
    for i in scored[:5]:
        indices.add(i)

    sample_msgs = [msgs[i] for i in sorted(indices)][:15]
    sample_lines = "\n".join(
        f"[{m['date']} {m['sender']}]: {m['text'][:200]}"
        for m in sample_msgs
    )

    prompt = (
        f"You are reviewing a WhatsApp chat history between {PRIMARY_USER_NAME} and "
        f"{contact}{' (' + relationship + ')' if relationship else ''}.\n\n"
        f"Date range: {date_from} to {date_to}\n"
        f"Total messages: {len(msgs):,}\n"
        f"Initiation: {PRIMARY_USER_NAME} starts {init.get('user_pct', '?')}% of conversations\n"
        f"Top topics: {', '.join(topics) if topics else 'not available'}\n"
        f"Distress signals detected: {ds_count}\n"
        f"Crisis flags detected: {cf_count}\n\n"
        f"Sample messages from across the timeline:\n{sample_lines}\n\n"
        "Write 2-3 paragraphs covering:\n"
        "1. The nature of this relationship and what it seems to mean to both parties\n"
        "2. Communication patterns — who drives the conversation, tone, frequency\n"
        "3. How the relationship appears to have evolved over the date range\n\n"
        "Write in plain prose, second person to the user. Be specific and grounded in what "
        "the data actually shows. Do not invent details not supported by the messages."
    )
    raw = _ollama_chat([{"role": "user", "content": prompt}], base_url, model, json_mode=False)
    return raw or ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _j(obj) -> str:
    """Serialize to JSON for inline JS, escaping </script> sequences."""
    return json.dumps(obj, ensure_ascii=False, default=str).replace("</script>", "<\\/script>")


def _escape_html(text: str) -> str:
    s = text or ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace('"', "&quot;").replace("'", "&#39;")
    return s


def _date_range(chats: list) -> str:
    all_dates = [m["date"] for c in chats for m in c["messages"] if m["date"]]
    if not all_dates:
        return ""
    return f"{min(all_dates)} – {max(all_dates)}"


class InteractiveTopologyGenerator:
    """Build entity-interaction network from people/ABSA data and render as Plotly JSON."""

    def __init__(self, people: dict):
        self.people = people

    def construct_interaction_graph(self):
        try:
            import networkx as nx
        except ImportError:
            return None
        G = nx.Graph()
        for name, data in self.people.items():
            dist = data.get("sentiment_dist", {})
            pos = dist.get("positive", 0) / 100
            neg = dist.get("negative", 0) / 100
            avg_v = data.get("avg_valence", pos - neg)
            G.add_node(name, avg_valence=avg_v, count=data.get("count", 1), is_contact=False)
            for chat_name in data.get("chats", []):
                if not G.has_node(chat_name):
                    G.add_node(chat_name, avg_valence=0.0, count=0, is_contact=True)
                if G.has_edge(chat_name, name):
                    G[chat_name][name]["weight"] += data.get("count", 1)
                    G[chat_name][name]["valences"].append(avg_v)
                else:
                    G.add_edge(chat_name, name, weight=data.get("count", 1), valences=[avg_v])
        return G

    def generate_plotly_figure(self, G) -> dict:
        try:
            import networkx as nx
        except ImportError:
            return {}
        if G is None or len(G.nodes) < 2:
            return {}
        pos = nx.spring_layout(G, seed=42, k=2.0)
        degree_centrality = nx.degree_centrality(G)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y, node_colors, node_sizes, node_hover = [], [], [], [], []
        for n in G.nodes():
            nd = G.nodes[n]
            avg_v = nd.get("avg_valence", 0.0)
            centrality = degree_centrality.get(n, 0.0)
            cnt = nd.get("count", 1)
            is_contact = nd.get("is_contact", False)
            if avg_v > 0.1:
                col = f"rgba(34,197,94,{min(0.9, 0.5 + abs(avg_v) * 0.4):.2f})"
            elif avg_v < -0.1:
                col = f"rgba(239,68,68,{min(0.9, 0.5 + abs(avg_v) * 0.4):.2f})"
            else:
                col = "rgba(156,163,175,0.7)"
            node_x.append(pos[n][0])
            node_y.append(pos[n][1])
            node_colors.append(col)
            node_sizes.append(int(20 + centrality * 40 + (12 if is_contact else 0)))
            node_hover.append(
                f"{n}<br>Valence: {avg_v:+.2f}<br>Centrality: {centrality:.2f}"
                f"<br>Mentions: {cnt}<br>{'(chat contact)' if is_contact else ''}"
            )

        return {
            "data": [
                {"type": "scatter", "x": edge_x, "y": edge_y, "mode": "lines",
                 "line": {"width": 1, "color": "rgba(156,163,175,0.4)"}, "hoverinfo": "none"},
                {"type": "scatter", "x": node_x, "y": node_y, "mode": "markers+text",
                 "marker": {"size": node_sizes, "color": node_colors,
                            "line": {"width": 1, "color": "rgba(0,0,0,0.1)"}},
                 "text": list(G.nodes()), "textposition": "top center",
                 "textfont": {"size": 11}, "hovertext": node_hover, "hoverinfo": "text"},
            ],
            "layout": {
                "showlegend": False, "hovermode": "closest",
                "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
                "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
                "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
                "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                "height": 520,
            },
        }


def generate_html(
    chats: list,
    cross_chat: dict,
    ai_results: dict,
    output_file: str,
) -> None:
    print(f"[INFO] Generating HTML → {output_file}")

    # Build per-chat display data
    chat_data = []
    for chat in chats:
        a = chat["analytics"]
        cd = {
            "id": re.sub(r"\W+", "_", chat["contact_name"]),
            "name": chat["contact_name"],
            "relationship": chat["contact_relationship"],
            "tags": chat["tags"],
            "has_framework": bool(chat.get("framework_content")),
            "framework_content": chat.get("framework_content") or "",
            "visit_tracker": a["visit_tracker"],
            "initiation_balance": a["initiation_balance"],
            "response_times": a["response_times"],
            "daily_sentiment": a["daily_sentiment"],
            "weekly_volume": a["weekly_volume"],
            "topics": a["topics"],
            "lda_topics": a.get("lda_topics", {}),
            "distress_signals": a["distress_signals"],
            "crisis_flags": a["crisis_flags"],
            "financial_requests": a["financial_requests"],
            "narrative_vs_record": ai_results.get("narrative_vs_record", {}).get(chat["contact_name"], []),
            "framework_suggestions": ai_results.get("framework_suggestions", {}).get(chat["contact_name"], []),
            "relationship_summary": ai_results.get("relationship_summary", {}).get(chat["contact_name"], ""),
            "message_count": len(chat["messages"]),
            "emotion_summary": a["emotion_summary"],
            "emotion_monthly": a["emotion_monthly"],
            "coping_summary": a["coping_summary"],
            "messages_for_timeline": [
                {
                    "date": str(m["date"]),
                    "sender": m["sender"],
                    "is_user": m["is_user"],
                    "text": m["text"][:300],
                    "sentiment": m.get("sentiment", {}).get("label", "neutral"),
                    "dominant_emotion": m.get("sentiment", {}).get("dominant_emotion", "neutral"),
                    "intents": [i for i in m.get("intents", []) if i not in ("INITIATED_BY_USER", "INITIATED_BY_CONTACT")],
                }
                for m in chat["messages"]
            ],
        }
        chat_data.append(cd)

    people_data = cross_chat.get("merged_people", {})
    people_ai = ai_results.get("people_cards", {})

    # Build relationship network topology
    topology_figure: dict = {}
    try:
        gen = InteractiveTopologyGenerator(people_data)
        G = gen.construct_interaction_graph()
        topology_figure = gen.generate_plotly_figure(G)
    except Exception as _topo_err:
        print(f"[WARN] Topology generation failed: {_topo_err}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhatsApp Analysis — {_escape_html(PRIMARY_USER_NAME)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.29.1/plotly.min.js"></script>
<style>
:root{{
  --bg:#f8f9fa;--surface:#ffffff;--border:#e2e8f0;--text:#1a202c;--muted:#718096;
  --user:#25D366;--contact:#1a73e8;--warn:#f59e0b;--danger:#ef4444;--ok:#10b981;
  --radius:8px;--shadow:0 1px 3px rgba(0,0,0,.1);
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.6}}
a{{color:var(--contact)}}
.header{{background:var(--surface);border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:var(--shadow)}}
.header h1{{font-size:18px;font-weight:700}}
.header .meta{{color:var(--muted);font-size:13px}}
.page-layout{{display:flex;align-items:flex-start}}
.sidebar{{width:220px;flex-shrink:0;background:var(--surface);border-right:1px solid var(--border);position:sticky;top:60px;height:calc(100vh - 60px);overflow-y:auto;padding:8px 0}}
.sidebar-title{{padding:8px 16px 6px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}}
.sb-btn{{display:block;width:100%;padding:10px 16px;border:none;background:transparent;text-align:left;cursor:pointer;border-left:3px solid transparent;transition:background .15s,border-color .15s}}
.sb-btn:hover{{background:var(--bg);border-left-color:var(--border)}}
.sb-btn.active{{background:#eff6ff;border-left-color:var(--contact)}}
.sb-btn.active .sb-name{{color:var(--contact)}}
.sb-btn.active .sb-sub,.sb-btn.active .sb-tag{{color:var(--contact);opacity:.8}}
.sb-btn.active .sb-flag{{color:var(--warn)}}
.sb-name{{display:block;font-weight:600;font-size:14px}}
.sb-sub{{display:block;font-size:12px;color:var(--muted);margin-top:2px}}
.sb-tag{{display:inline-block;font-size:10px;padding:1px 6px;border-radius:8px;background:var(--bg);border:1px solid var(--border);margin:3px 2px 0 0}}
.sb-flag{{display:inline-block;font-size:11px;color:var(--warn);margin-top:3px}}
.main-area{{flex:1;min-width:0;display:flex;flex-direction:column}}
.tabs{{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex;gap:0;overflow-x:auto;position:sticky;top:60px;z-index:99;box-shadow:var(--shadow)}}
.tab-btn{{padding:12px 18px;border:none;background:transparent;cursor:pointer;font-size:14px;font-weight:500;color:var(--muted);border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s}}
.tab-btn.active{{color:var(--contact);border-bottom-color:var(--contact)}}
.main{{padding:24px}}
.panel{{display:none}}.panel.active{{display:block}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px;box-shadow:var(--shadow)}}
.card h2{{font-size:16px;font-weight:700;margin-bottom:12px;color:var(--text)}}
.card h3{{font-size:14px;font-weight:600;margin-bottom:8px;color:var(--text)}}
.stat-row{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}}
.stat{{background:var(--bg);border-radius:var(--radius);padding:14px 18px;flex:1;min-width:120px;text-align:center}}
.stat .val{{font-size:28px;font-weight:700}}
.stat .lbl{{font-size:12px;color:var(--muted);margin-top:4px}}
.stat.ok .val{{color:var(--ok)}}
.stat.warn .val{{color:var(--warn)}}
.stat.danger .val{{color:var(--danger)}}
.bubble-day{{margin-bottom:20px}}
.day-label{{font-size:12px;color:var(--muted);text-align:center;margin-bottom:8px;font-weight:600}}
.bubble{{display:flex;margin-bottom:6px}}
.bubble.user{{justify-content:flex-end}}
.bubble-inner{{max-width:65%;padding:10px 14px;border-radius:18px;font-size:14px;line-height:1.5;position:relative}}
.bubble.user .bubble-inner{{background:var(--user);color:#fff;border-bottom-right-radius:4px}}
.bubble.contact .bubble-inner{{background:var(--surface);border:1px solid var(--border);border-bottom-left-radius:4px}}
.bubble .badge{{display:inline-block;font-size:10px;padding:2px 6px;border-radius:10px;margin-left:6px;vertical-align:middle}}
.badge.distress{{background:#fef3c7;color:#92400e}}
.badge.crisis{{background:#fee2e2;color:#991b1b}}
.badge.visit_offered{{background:#dbeafe;color:#1e40af}}
.badge.visit_cancelled{{background:#fed7aa;color:#9a3412}}
.badge.visit_happened{{background:#dcfce7;color:#166534}}
.badge.financial{{background:#f3e8ff;color:#6b21a8}}
.bar{{height:20px;border-radius:4px;display:flex;overflow:hidden;margin-bottom:8px}}
.bar-user{{background:var(--user)}}
.bar-contact{{background:var(--contact)}}
.bar-label{{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:2px}}
.timeline-event{{padding:10px 14px;border-left:3px solid var(--border);margin-bottom:8px;font-size:13px}}
.timeline-event.positive{{border-color:var(--ok)}}
.timeline-event.warning{{border-color:var(--warn)}}
.timeline-event.crisis{{border-color:var(--danger)}}
.timeline-event .event-date{{font-size:11px;color:var(--muted);margin-bottom:4px}}
.person-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:12px}}
.person-card .name{{font-weight:700;font-size:15px;margin-bottom:4px}}
.person-card .summary{{color:var(--text);font-size:13px;margin-bottom:8px}}
.person-card .label{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;background:var(--bg);border:1px solid var(--border);margin-bottom:8px}}
.person-card .sentiment-bar{{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:4px}}
.person-card .sb-pos{{background:var(--ok)}}
.person-card .sb-neu{{background:var(--border)}}
.person-card .sb-neg{{background:var(--danger)}}
.contrast-pair{{border:1px solid var(--border);border-radius:var(--radius);padding:14px;margin-bottom:10px}}
.contrast-pair .claimed{{color:var(--muted);font-size:13px;margin-bottom:4px}}
.contrast-pair .claimed::before{{content:"💬 Claimed: ";font-weight:600}}
.contrast-pair .record{{color:var(--text);font-size:13px}}
.contrast-pair .record::before{{content:"📊 Record shows: ";font-weight:600}}
.fw-item{{border-left:3px solid var(--contact);padding:8px 12px;margin-bottom:10px;font-size:13px}}
.fw-item .fw-original{{font-weight:600;margin-bottom:4px}}
.fw-item .fw-suggestion{{color:var(--contact);margin-bottom:2px}}
.fw-item .fw-basis{{color:var(--muted);font-size:12px;font-style:italic}}
.visit-event{{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);font-size:13px}}
.visit-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.visit-dot.offered{{background:var(--contact)}}
.visit-dot.accepted{{background:var(--ok)}}
.visit-dot.cancelled{{background:var(--warn)}}
.visit-dot.happened{{background:var(--ok)}}
.reveal-btn{{background:var(--danger);color:#fff;border:none;padding:10px 20px;border-radius:var(--radius);cursor:pointer;font-size:14px;margin-bottom:16px}}
.crisis-content{{display:none}}
.crisis-safe{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:var(--radius);padding:14px;font-size:14px;margin-top:16px}}
.no-data{{color:var(--muted);font-size:14px;text-align:center;padding:32px}}
.chart-toggle-bar{{display:flex;gap:6px;margin-bottom:10px}}
.chart-toggle{{padding:4px 14px;border:1.5px solid var(--border);border-radius:20px;font-size:12px;cursor:pointer;background:var(--surface);color:var(--muted);transition:all .15s}}
.chart-toggle.active{{background:var(--contact);color:#fff;border-color:var(--contact)}}
.notes-toolbar{{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;margin-bottom:16px;box-shadow:var(--shadow)}}
.btn-note-action{{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;color:var(--text)}}
.btn-note-action:hover{{background:var(--contact);color:#fff;border-color:var(--contact)}}
.note-section{{margin-top:16px;border-top:2px dashed #e2e8f0;padding-top:12px}}
.note-label{{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:7px}}
textarea.note{{width:100%;border:2px dashed #d1d5db;border-radius:var(--radius);padding:10px 12px;font-size:13px;resize:vertical;min-height:80px;background:#fefce8;color:var(--text);font-family:inherit;line-height:1.6;transition:border-color .15s,background .15s;outline:none}}
textarea.note:focus{{border-color:#f59e0b;background:#fff}}
textarea.note:not(:placeholder-shown){{background:#fffbeb;border-color:#d97706;border-style:solid}}
.chart-wrap{{position:relative;height:280px;margin-bottom:16px}}
.chart-wrap-sm{{position:relative;height:200px;margin-bottom:8px}}
.tag{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:var(--bg);border:1px solid var(--border);margin-right:4px}}
.cross-chat-card{{border-left:4px solid var(--warn);padding:12px 16px;background:var(--surface);border-radius:0 var(--radius) var(--radius) 0;margin-bottom:10px;font-size:13px}}
@media(max-width:700px){{
  .page-layout{{flex-direction:column}}
  .sidebar{{width:100%;height:auto;position:static;border-right:none;border-bottom:1px solid var(--border);display:flex;flex-wrap:wrap;gap:6px;padding:10px 16px}}
  .sidebar-title{{display:none}}
  .sb-btn{{width:auto;display:inline-block;padding:5px 12px;border-radius:20px;border:1px solid var(--border);border-left:1px solid var(--border);font-size:13px;margin:0}}
  .sb-btn:hover{{background:var(--bg);border-left-color:var(--border)}}
  .sb-btn.active{{background:var(--contact);color:#fff;border-left-color:var(--contact)}}
  .sb-btn.active .sb-name{{color:#fff}}
  .sb-sub,.sb-tag,.sb-flag{{display:none}}
  .sb-name{{font-size:13px}}
  .stat-row{{gap:8px}}.stat{{min-width:90px;padding:10px}}
  .bubble-inner{{max-width:82%}}
  .header h1{{font-size:15px}}
}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>WhatsApp Analysis — {_escape_html(PRIMARY_USER_NAME)}</h1>
    <div class="meta">{_escape_html(_date_range(chats))} &middot; {sum(len(c["messages"]) for c in chats):,} messages across {len(chats)} chat{"s" if len(chats) != 1 else ""}</div>
  </div>
</div>

<div class="page-layout">
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-title">Conversations</div>
  </aside>
  <div class="main-area">
    <div id="tabBar" class="tabs"></div>
    <div class="main" id="mainContent"></div>
  </div>
</div>

<script>
const PRIMARY_USER = {_j(PRIMARY_USER_NAME)};
const CHATS = {_j(chat_data)};
const CROSS_CHAT = {_j(cross_chat)};
const PEOPLE = {_j(people_data)};
const PEOPLE_AI = {_j(people_ai)};
const TOPOLOGY_FIGURE = {_j(topology_figure)};
const TIMELINE_HIGHLIGHTS = {_j(ai_results.get("timeline_highlights", []))};
const CROSS_CHAT_LINKS = {_j(ai_results.get("cross_chat_links", []))};
const EMOTION_COLORS = {{
  anger:"#ef4444",disgust:"#a855f7",fear:"#f97316",joy:"#22c55e",
  neutral:"#9ca3af",sadness:"#3b82f6",surprise:"#eab308"
}};
const EMOTION_LABELS = ["anger","disgust","fear","joy","neutral","sadness","surprise"];
const CRISIS_ASSESSED = {_j(ai_results.get("crisis_assessed", {}))};
const COPING_AI = {_j(ai_results.get("coping_analysis", {}))};

let currentChat = null;
let currentTab = "conversations";
let _lastSingleTab = "timeline";
let charts = {{}};

// ── Utilities ────────────────────────────────────────────────────────────────
function esc(s) {{
  return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}
function fmt(dateStr) {{
  if(!dateStr||dateStr==="None"||dateStr==="null") return "";
  const d = new Date(dateStr); if(isNaN(d)) return dateStr;
  return d.toLocaleDateString("en-GB",{{day:"numeric",month:"short",year:"numeric"}});
}}
function badge(intent) {{
  const map={{
    DISTRESS_SIGNAL:"distress",CRISIS_FLAG:"crisis",VISIT_OFFERED:"visit_offered",
    VISIT_CANCELLED:"visit_cancelled",VISIT_HAPPENED:"visit_happened",
    FINANCIAL_REQUEST:"financial"
  }};
  const labels={{
    DISTRESS_SIGNAL:"distress",CRISIS_FLAG:"⚠ crisis",VISIT_OFFERED:"visit offered",
    VISIT_CANCELLED:"visit cancelled",VISIT_HAPPENED:"✓ visit",
    FINANCIAL_REQUEST:"financial"
  }};
  const cls=map[intent]||"";
  return cls ? `<span class="badge ${{cls}}">${{labels[intent]||intent}}</span>` : "";
}}

// ── Note persistence ─────────────────────────────────────────────────────────
// Primary: localStorage (works for standalone HTML in any browser)
// Secondary: pywebview API (writes to ~/.whatsapp_analyser_notes.json when
//            running inside the desktop app, survives HTML regeneration)
function noteKey(chatId, date) {{ return `note_${{chatId}}_${{date}}`; }}

function saveNote(chatId, date, val) {{
  const key = noteKey(chatId, date);
  localStorage.setItem(key, val);
  if(typeof pywebview !== 'undefined' && pywebview.api && pywebview.api.save_note)
    pywebview.api.save_note(key, val).catch(()=>{{}});
}}
function loadNote(chatId, date) {{
  return localStorage.getItem(noteKey(chatId, date)) || "";
}}

function personEditKey(name) {{ return `person_edit_${{name}}`; }}
function savePersonEdit(name, val) {{
  localStorage.setItem(personEditKey(name), val);
  if(typeof pywebview !== 'undefined' && pywebview.api && pywebview.api.save_note)
    pywebview.api.save_note(personEditKey(name), val).catch(()=>{{}});
}}
function loadPersonEdit(name) {{ return localStorage.getItem(personEditKey(name)) || ""; }}

function exportNotes() {{
  const out = {{}};
  for(let i = 0; i < localStorage.length; i++) {{
    const k = localStorage.key(i);
    if(k && (k.startsWith('note_') || k.startsWith('person_edit_')))
      out[k] = localStorage.getItem(k);
  }}
  if(!Object.keys(out).length) {{ alert('No notes saved yet.'); return; }}
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}}));
  a.download = 'whatsapp_analyser_notes.json';
  a.click();
}}
function importNotes() {{
  const inp = document.createElement('input');
  inp.type = 'file'; inp.accept = '.json';
  inp.onchange = e => {{
    const reader = new FileReader();
    reader.onload = ev => {{
      try {{
        const notes = JSON.parse(ev.target.result);
        Object.entries(notes).forEach(([k,v]) => {{
          localStorage.setItem(k, v);
          if(typeof pywebview !== 'undefined' && pywebview.api && pywebview.api.save_note)
            pywebview.api.save_note(k, v).catch(()=>{{}});
        }});
        renderMain();
      }} catch(err) {{ alert('Could not read notes file: ' + err.message); }}
    }};
    reader.readAsText(e.target.files[0]);
  }};
  inp.click();
}}

// ── Chart helpers ─────────────────────────────────────────────────────────
function destroyChart(id) {{
  if(charts[id]) {{ charts[id].destroy(); delete charts[id]; }}
}}
function makeLineChart(canvasId, labels, datasets, isSentiment) {{
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if(!ctx) return;
  const opts = {{
    responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:"top"}},tooltip:{{mode:"index"}}}},
    scales:{{
      x:{{ticks:{{maxTicksLimit:12,maxRotation:45}}}},
      y:{{min:-1,max:1,ticks:{{stepSize:0.5}}}}
    }}
  }};
  if(isSentiment) {{
    opts.onClick = function(evt, elements) {{
      if(!elements||!elements.length) return;
      const idx = elements[0].index;
      const label = labels[idx];
      if(label) showMoodExplainer(canvasId, label);
    }};
    opts.onHover = function(evt) {{
      evt.native.target.style.cursor = 'pointer';
    }};
  }}
  charts[canvasId] = new Chart(ctx,{{
    type:"line",
    data:{{labels,datasets}},
    options:opts
  }});
}}
function makeBarChart(canvasId, labels, datasets, opts) {{
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if(!ctx) return;
  charts[canvasId] = new Chart(ctx,{{
    type:"bar",
    data:{{labels,datasets}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{position:"top"}},tooltip:{{mode:"index"}}}},
      scales:{{
        x:{{stacked:false,ticks:{{maxTicksLimit:16,maxRotation:45}}}},
        y:{{beginAtZero:true,...(opts||{{}})}}
      }}
    }}
  }});
}}

// ── Sentiment helpers ─────────────────────────────────────────────────────
function dateToIsoWeek(dateStr) {{
  var d = new Date(dateStr + 'T00:00:00');
  var day = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - day);
  var y = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return d.getUTCFullYear() + '-W' + String(Math.ceil(((d - y) / 86400000 + 1) / 7)).padStart(2, '0');
}}
// Weekly average of abs(score) — "how emotionally charged was this week?"
function absWeekly(dailySeries) {{
  var wm = {{}};
  Object.entries(dailySeries).forEach(function(kv) {{
    var w = dateToIsoWeek(kv[0]);
    if(!wm[w]) wm[w] = [];
    wm[w].push(Math.abs(kv[1]));
  }});
  var weeks = Object.keys(wm).sort();
  return {{ weeks: weeks, vals: weeks.map(function(w) {{ return wm[w].reduce(function(a,b){{return a+b;}},0) / wm[w].length; }}) }};
}}
// Daily difference: user score minus contact score
// Positive = you were more positive than them; negative = they were more positive
function dailyDiff(userSeries, contactSeries) {{
  var diff = {{}};
  Object.keys(userSeries).forEach(function(d) {{
    if(contactSeries[d] !== undefined) diff[d] = userSeries[d] - contactSeries[d];
  }});
  return diff;
}}

// ── Chat data helpers ────────────────────────────────────────────────────
function getChatById(id) {{ return CHATS.find(c=>c.id===id)||null; }}
function getAllDates(chats) {{
  const s=new Set(); chats.forEach(c=>c.messages_for_timeline.forEach(m=>{{if(m.date)s.add(m.date)}}));
  return [...s].sort();
}}
function rollingAvg(obj, window=7) {{
  const keys=Object.keys(obj).sort();
  const vals=keys.map(k=>obj[k]);
  const out={{}};
  keys.forEach((k,i)=>{{
    const slice=vals.slice(Math.max(0,i-window+1),i+1);
    out[k]=slice.reduce((a,b)=>a+b,0)/slice.length;
  }});
  return out;
}}

// ── Tab definitions ─────────────────────────────────────────────────────
function getTabsForChat(chatId) {{
  const base=[
    {{id:"conversations",label:"Conversations",showAll:true,showSingle:false}},
    {{id:"timeline",label:"Timeline",showAll:true,showSingle:true}},
    {{id:"visits",label:"Visits & Availability",showAll:false,showSingle:true}},
    {{id:"sentiment",label:"Sentiment",showAll:true,showSingle:true}},
    {{id:"initiation",label:"Initiation Balance",showAll:true,showSingle:true}},
    {{id:"crisis",label:"⚠ Crisis Moments",showAll:true,showSingle:true}},
    {{id:"support",label:"Support Given",showAll:true,showSingle:true}},
    {{id:"coping",label:"Coping Dynamics",showAll:true,showSingle:true}},
    {{id:"people",label:"People & Sentiment",showAll:true,showSingle:true}},
    {{id:"topics",label:"Topics Over Time",showAll:true,showSingle:true}},
    {{id:"network",label:"Relationship Network",showAll:true,showSingle:true}},
    {{id:"narrative",label:"Narrative vs Record",showAll:false,showSingle:true}},
    {{id:"summary",label:"Relationship Summary",showAll:false,showSingle:true}},
    {{id:"ask",label:"Ask AI",showAll:true,showSingle:true}},
  ];
  const isAll = chatId==="all";
  let tabs=base.filter(t=>isAll?t.showAll:t.showSingle);
  // Framework tab: only show for single chat if it has a framework
  if(!isAll) {{
    const chat=getChatById(chatId);
    if(chat&&chat.has_framework) tabs.push({{id:"framework",label:"Communications Framework",showAll:false,showSingle:true}});
  }}
  return tabs;
}}

function renderTabBar(chatId) {{
  const tabs=getTabsForChat(chatId);
  const bar=document.getElementById("tabBar");
  bar.innerHTML=tabs.map(t=>
    `<button class="tab-btn${{currentTab===t.id?" active":""}}" onclick="selectTab('${{t.id}}')">${{t.label}}</button>`
  ).join("");
}}

// ── Main render dispatcher ───────────────────────────────────────────────
function buildSidebar() {{
  const sb=document.getElementById("sidebar");
  if(!sb) return;
  sb.querySelectorAll(".sb-btn").forEach(b=>b.remove());
  if(CHATS.length>1) {{
    const b=document.createElement("button");
    b.className="sb-btn"; b.dataset.chat="all";
    b.innerHTML=`<span class="sb-name">All conversations</span><span class="sb-sub">${{CHATS.length}} chats</span>`;
    b.onclick=()=>selectChat(b); sb.appendChild(b);
  }}
  CHATS.forEach(chat=>{{
    const b=document.createElement("button");
    b.className="sb-btn"; b.dataset.chat=chat.id;
    const tags=(chat.tags||[]).map(t=>`<span class="sb-tag">${{esc(t)}}</span>`).join("");
    const dc=(chat.distress_signals||[]).length;
    b.innerHTML=`<span class="sb-name">${{esc(chat.name)}}</span>`+
      (chat.relationship?`<span class="sb-sub">${{esc(chat.relationship)}}</span>`:"")+
      `<div style="margin-top:3px">${{tags}}${{dc?`<span class="sb-flag">⚠ ${{dc}}</span>`:""}}</div>`;
    b.onclick=()=>selectChat(b); sb.appendChild(b);
  }});
}}

function selectChat(btn) {{
  document.querySelectorAll(".sb-btn").forEach(b=>b.classList.remove("active"));
  btn.classList.add("active");
  currentChat=btn.dataset.chat;
  const availTabs=getTabsForChat(currentChat).map(t=>t.id);
  if(!availTabs.includes(currentTab)) {{
    if(currentChat==="all") {{
      currentTab="conversations";
    }} else {{
      currentTab=availTabs.includes(_lastSingleTab)?_lastSingleTab:"timeline";
    }}
  }}
  renderTabBar(currentChat);
  renderMain();
}}

function selectTab(tabId) {{
  currentTab=tabId;
  if(currentChat!=="all") _lastSingleTab=tabId;
  renderTabBar(currentChat);
  renderMain();
}}

function renderMain() {{
  const el=document.getElementById("mainContent");
  if(!currentChat) {{el.innerHTML="";return;}}
  if(currentChat==="all") {{
    renderAllView(el);
  }} else {{
    const chat=getChatById(currentChat);
    if(!chat) {{el.innerHTML=`<p class="no-data">Chat not found.</p>`;return;}}
    switch(currentTab) {{
      case "timeline": renderTimeline(el,chat); break;
      case "visits": renderVisits(el,chat); break;
      case "sentiment": renderSentimentChart(el,[chat]); break;
      case "initiation": renderInitiation(el,[chat]); break;
      case "crisis": renderCrisis(el,chat); break;
      case "support": renderSupport(el,[chat]); break;
      case "coping": renderCoping(el,[chat]); break;
      case "people": renderPeople(el); break;
      case "topics": renderTopics(el,[chat]); break;
      case "network": renderNetwork(el); break;
      case "framework": renderFramework(el,chat); break;
      case "narrative": renderNarrative(el,chat); break;
      case "summary": renderSummary(el,chat); break;
      case "ask": renderAsk(el,chat.id); break;
      default: renderTimeline(el,chat);
    }}
  }}
}}

// ── Conversations (All view) ─────────────────────────────────────────────
function renderAllView(el) {{
  switch(currentTab) {{
    case "conversations": renderConversationsPanel(el); break;
    case "timeline": renderMergedTimeline(el); break;
    case "sentiment": renderSentimentChart(el,CHATS); break;
    case "initiation": renderInitiation(el,CHATS); break;
    case "crisis": renderAllCrisis(el); break;
    case "support": renderSupport(el,CHATS); break;
    case "coping": renderCoping(el,CHATS); break;
    case "people": renderPeople(el); break;
    case "topics": renderTopics(el,CHATS); break;
    case "network": renderNetwork(el); break;
    case "ask": renderAsk(el,"all"); break;
    default: renderConversationsPanel(el);
  }}
}}

function renderConversationsPanel(el) {{
  let html=`<div class="card"><h2>Conversations overview</h2>`;
  CHATS.forEach(chat=>{{
    const init=chat.initiation_balance;
    const visits=chat.visit_tracker;
    html+=`
    <div style="border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:start;flex-wrap:wrap;gap:8px">
        <div>
          <b>${{esc(chat.name)}}</b>
          ${{chat.tags.map(t=>`<span class="tag">${{t}}</span>`).join("")}}
          <div style="color:var(--muted);font-size:12px">${{esc(chat.relationship)}} &middot; ${{chat.message_count.toLocaleString()}} messages</div>
        </div>
        <div style="font-size:13px;color:var(--muted)">
          Distress signals: <b>${{chat.distress_signals.length}}</b> &nbsp;
          Crisis flags: <b>${{chat.crisis_flags.length}}</b>
        </div>
      </div>
      <div style="margin-top:12px">
        <div class="bar-label"><span>Initiation: You ${{init.user_pct}}%</span><span>Them ${{init.contact_pct}}%</span></div>
        <div class="bar"><div class="bar-user" style="width:${{init.user_pct}}%"></div><div class="bar-contact" style="width:${{init.contact_pct}}%"></div></div>
      </div>
      <div style="font-size:13px;margin-top:8px">
        Visits: ${{visits.offered}} offered &nbsp; ${{visits.happened}} happened &nbsp; ${{visits.cancelled}} cancelled
      </div>
    </div>`;
  }});
  html+=`</div>`;

  // Cross-chat links
  if(CROSS_CHAT_LINKS&&CROSS_CHAT_LINKS.length) {{
    html+=`<div class="card"><h2>Cross-chat pattern links</h2>`;
    CROSS_CHAT_LINKS.forEach(l=>{{
      html+=`<div class="cross-chat-card">
        <div style="font-weight:600;margin-bottom:4px">${{esc(l.date_range)}} — ${{esc(l.chats_involved)}}</div>
        <div>${{esc(l.pattern)}}</div>
        <div style="color:var(--muted);font-size:12px;margin-top:4px">${{esc(l.significance)}}</div>
      </div>`;
    }});
    html+=`</div>`;
  }}

  // Shared people
  if(CROSS_CHAT.shared_people&&CROSS_CHAT.shared_people.length) {{
    html+=`<div class="card"><h2>People mentioned across multiple chats</h2>`;
    CROSS_CHAT.shared_people.forEach(sp=>{{
      html+=`<div style="padding:8px 0;border-bottom:1px solid var(--border);font-size:13px">
        <b>${{esc(sp.name)}}</b> — mentioned in: ${{sp.chats.map(c=>`<i>${{esc(c)}}</i>`).join(", ")}}
      </div>`;
    }});
    html+=`</div>`;
  }}

  // Timeline highlights
  if(TIMELINE_HIGHLIGHTS&&TIMELINE_HIGHLIGHTS.length) {{
    html+=`<div class="card"><h2>Key timeline moments</h2>`;
    TIMELINE_HIGHLIGHTS.forEach(h=>{{
      html+=`<div class="timeline-event ${{esc(h.type)}}">
        <div class="event-date">${{fmt(h.date)}} — ${{esc(h.chat)}}</div>
        <div>${{esc(h.significance)}}</div>
      </div>`;
    }});
    html+=`</div>`;
  }}

  el.innerHTML=html;
}}

// ── Timeline ──────────────────────────────────────────────────────────────
function renderTimeline(el, chat) {{
  const msgs=chat.messages_for_timeline;
  // Notes toolbar — always visible at top of timeline
  el.innerHTML=`<div class="notes-toolbar">
    <span style="font-size:12px;color:var(--muted)">📝 Notes are saved in your browser. Back them up to keep them safe.</span>
    <div style="display:flex;gap:8px">
      <button class="btn-note-action" onclick="exportNotes()">⬇ Export notes</button>
      <button class="btn-note-action" onclick="importNotes()">⬆ Import notes</button>
    </div>
  </div>`;
  // Group by date
  const byDate={{}};
  msgs.forEach(m=>{{if(m.date){{(byDate[m.date]||(byDate[m.date]=[])).push(m);}} }});
  const dates=Object.keys(byDate).sort();
  let html=`<div id="timeline-wrap">`;
  dates.forEach(date=>{{
    const dayMsgs=byDate[date];
    const intentsToday=[...new Set(dayMsgs.flatMap(m=>m.intents))];
    const hasNote=loadNote(chat.id,date)!=="";
    html+=`<div class="bubble-day">
      <div class="day-label">${{fmt(date)}}${{intentsToday.map(i=>badge(i)).join("")}}</div>`;
    dayMsgs.forEach(m=>{{
      const side=m.is_user?"user":"contact";
      const emo=m.dominant_emotion&&m.dominant_emotion!=="neutral"?m.dominant_emotion:null;
      const emoDot=emo?`<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${{EMOTION_COLORS[emo]||"transparent"}};margin-left:5px;vertical-align:middle;opacity:.85" title="${{emo}}"></span>`:"";
      html+=`<div class="bubble ${{side}}">
        <div class="bubble-inner">${{esc(m.text)}}${{emoDot}}</div>
      </div>`;
    }});
    html+=`<div class="note-section">
      <div class="note-label"><span>📝</span> Your private notes — saved in this browser only</div>
      <textarea class="note" placeholder="Add context, reflections, or anything worth remembering about this day…" onchange="saveNote('${{chat.id}}','${{date}}',this.value)">${{esc(loadNote(chat.id,date))}}</textarea>
    </div>`;
    html+=`</div>`;
  }});
  html+=`</div>`;
  el.innerHTML += html;
}}

function renderMergedTimeline(el) {{
  // Merge all chats
  const allMsgs=[];
  CHATS.forEach(chat=>{{
    chat.messages_for_timeline.forEach(m=>allMsgs.push({{...m,chat_name:chat.name,chat_id:chat.id}}));
  }});
  allMsgs.sort((a,b)=>a.date<b.date?-1:a.date>b.date?1:0);
  const byDate={{}};
  allMsgs.forEach(m=>{{(byDate[m.date]||(byDate[m.date]=[])).push(m);}});
  const colors=["#1a73e8","#e91e63","#ff9800","#9c27b0","#00bcd4"];
  const chatColors={{}};
  CHATS.forEach((c,i)=>{{chatColors[c.id]=colors[i%colors.length];}});
  let html=`<div>`;
  Object.keys(byDate).sort().forEach(date=>{{
    const msgs=byDate[date];
    const byChat={{}};
    msgs.forEach(m=>{{(byChat[m.chat_id]||(byChat[m.chat_id]=[])).push(m);}});
    html+=`<div class="bubble-day"><div class="day-label">${{fmt(date)}}</div>`;
    Object.entries(byChat).forEach(([chatId,cMsgs])=>{{
      const chat=getChatById(chatId);
      html+=`<div style="border-left:3px solid ${{chatColors[chatId]}};padding-left:10px;margin-bottom:8px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${{esc(chat?chat.name:chatId)}}</div>`;
      cMsgs.slice(0,5).forEach(m=>{{
        html+=`<div class="bubble ${{m.is_user?"user":"contact"}}">
          <div class="bubble-inner" style="${{m.is_user?"":"border-left:3px solid "+chatColors[chatId]}}">${{esc(m.text.slice(0,200))}}</div>
        </div>`;
      }});
      if(cMsgs.length>5) html+=`<div style="color:var(--muted);font-size:12px;text-align:center">+${{cMsgs.length-5}} more messages</div>`;
      html+=`</div>`;
    }});
    html+=`</div>`;
  }});
  html+=`</div>`;
  el.innerHTML=html;
}}

// ── Visits ────────────────────────────────────────────────────────────────
function renderVisits(el, chat) {{
  const v=chat.visit_tracker;
  let html=`<div class="card"><h2>Visits & Availability — ${{esc(chat.name)}}</h2>
    <div class="stat-row">
      <div class="stat"><div class="val" style="color:var(--contact)">${{v.offered}}</div><div class="lbl">Offered</div></div>
      <div class="stat ok"><div class="val">${{v.happened}}</div><div class="lbl">Happened</div></div>
      <div class="stat warn"><div class="val">${{v.cancelled}}</div><div class="lbl">Cancelled</div></div>
      <div class="stat"><div class="val">${{v.accepted}}</div><div class="lbl">Accepted</div></div>
    </div>`;
  if(Object.keys(v.cancellations_by).length) {{
    html+=`<div style="font-size:13px;margin-bottom:12px">Cancellations by: `;
    Object.entries(v.cancellations_by).forEach(([who,n])=>{{
      html+=`<b>${{esc(who)}}</b>: ${{n}} &nbsp; `;
    }});
    html+=`</div>`;
  }}
  if(v.events&&v.events.length) {{
    html+=`<h3>Event log</h3>`;
    v.events.forEach(e=>{{
      html+=`<div class="visit-event">
        <div class="visit-dot ${{e.type}}"></div>
        <div style="color:var(--muted);font-size:12px;min-width:90px">${{fmt(e.date)}}</div>
        <div><b style="text-transform:capitalize">${{e.type}}</b> by ${{esc(e.sender)}}</div>
      </div>`;
    }});
  }} else {{
    html+=`<p class="no-data">No visit events detected.</p>`;
  }}
  html+=`</div>`;
  el.innerHTML=html;
  el.scrollTop=0;
}}

// ── Sentiment chart ────────────────────────────────────────────────────────
// ── Stacked chart toggle helper ───────────────────────────────────────────
// Used by both Emotion over Time and Topics Over Time charts.
// Stores the raw series data on the button bar so we can re-render on toggle.
function _buildStackedDatasets(series, colorMap, months) {{
  return Object.keys(series).map(function(key) {{
    const color = colorMap[key] || "#9ca3af";
    return {{
      label: key,
      data: series[key],
      backgroundColor: color + "cc",
      borderColor: color,
      borderWidth: 1.5,
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }};
  }});
}}

function _build100Datasets(series, colorMap, months) {{
  const totals = months.map((_,i) =>
    Object.values(series).reduce((s, arr) => s + (arr[i] || 0), 0) || 1
  );
  return Object.keys(series).map(function(key) {{
    const color = colorMap[key] || "#9ca3af";
    return {{
      label: key,
      data: series[key].map((v,i) => Math.round(v / totals[i] * 100)),
      backgroundColor: color + "cc",
      borderColor: color,
      borderWidth: 1,
      fill: true,
    }};
  }});
}}

function drawToggleChart(canvasId, series, colorMap, months, mode) {{
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const datasets = mode === "stack"
    ? _build100Datasets(series, colorMap, months)
    : _buildStackedDatasets(series, colorMap, months);
  const isStack = mode === "stack";
  charts[canvasId] = new Chart(ctx, {{
    type: isStack ? "bar" : "line",
    data: {{ labels: months, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        x: {{ stacked: true, ticks: {{ font: {{ size: 11 }} }} }},
        y: {{
          stacked: true,
          ticks: {{ font: {{ size: 11 }}, callback: isStack ? (v=>v+"%") : undefined }},
          max: isStack ? 100 : undefined,
        }},
      }},
      plugins: {{ legend: {{ position: "bottom", labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }} }},
    }},
  }});
}}

function switchChartMode(canvasId, mode, btn) {{
  btn.closest(".chart-toggle-bar").querySelectorAll(".chart-toggle")
    .forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const wrap = btn.closest(".chart-section-wrap");
  const series = JSON.parse(wrap.dataset.series);
  const colorMap = JSON.parse(wrap.dataset.colormap);
  const months = JSON.parse(wrap.dataset.months);
  requestAnimationFrame(() => drawToggleChart(canvasId, series, colorMap, months, mode));
}}

function renderSentimentChart(el, chats) {{
  el.innerHTML=`
    <div class="card">
      <h2>Mood over time</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:14px">
        Score runs from <b>−1</b> (most negative) through <b>0</b> (neutral) to <b>+1</b> (most positive).
        Plotted as a 7-day rolling average to smooth day-to-day noise.
        Solid line = you &nbsp;|&nbsp; Dashed line = contact.
      </p>
      <div class="chart-wrap"><canvas id="sentChart"></canvas></div>
    </div>
    <div class="card">
      <h2>Message volume</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:14px">
        Number of messages sent per week by each person.
        Gaps or sudden drops often indicate a period of silence or conflict.
      </p>
      <div class="chart-wrap-sm"><canvas id="volChart"></canvas></div>
    </div>`;

  const colors=["#25D366","#1a73e8","#e91e63","#ff9800","#9c27b0"];
  const colorsDash=["#25D366cc","#1a73e8cc","#e91e63cc","#ff9800cc","#9c27b0cc"];

  // ── Mood chart (with crisis/distress overlays) ───────────────────────────
  // Extend allDates to include any crisis/distress dates not already present
  const _flagDates=chats.flatMap(c=>
    [...(c.crisis_flags||[]),...(c.distress_signals||[])].map(s=>s.date).filter(Boolean)
  );
  const allDates=[...new Set([
    ...chats.flatMap(c=>Object.keys(c.daily_sentiment.user||{{}}).concat(Object.keys(c.daily_sentiment.contact||{{}}))),
    ..._flagDates
  ])].sort();

  const sentDatasets=[];
  chats.forEach((chat,i)=>{{
    const userAvg=rollingAvg(chat.daily_sentiment.user||{{}});
    const contAvg=rollingAvg(chat.daily_sentiment.contact||{{}});
    const label=chats.length>1?chat.name:"";

    // Mood lines
    sentDatasets.push({{
      label:label?`You (${{label}})`:"You",
      data:allDates.map(d=>userAvg[d]??null),
      borderColor:colors[i%colors.length],backgroundColor:colors[i%colors.length]+"22",
      borderWidth:2,pointRadius:1,spanGaps:true,tension:.3
    }});
    sentDatasets.push({{
      label:label?chat.name:chat.name,
      data:allDates.map(d=>contAvg[d]??null),
      borderColor:colorsDash[i%colorsDash.length],backgroundColor:"transparent",
      borderWidth:2,borderDash:[4,4],pointRadius:1,spanGaps:true,tension:.3
    }});

    // Distress signal markers — orange dots near the bottom of the chart
    const distressMap={{}};
    (chat.distress_signals||[]).forEach(s=>{{
      if(s.date) (distressMap[s.date]||(distressMap[s.date]=[])).push(s.text.slice(0,140));
    }});
    if(Object.keys(distressMap).length) {{
      sentDatasets.push({{
        label:label?`Distress (${{label}})`:"Distress signal",
        data:allDates.map(d=>distressMap[d]?-0.87:null),
        _tips:distressMap,
        backgroundColor:"rgba(245,158,11,.9)",borderColor:"rgba(245,158,11,1)",
        pointStyle:"circle",pointRadius:8,pointHoverRadius:11,
        showLine:false,spanGaps:false
      }});
    }}

    // Crisis flag markers — red triangles at the very bottom
    const crisisMap={{}};
    (chat.crisis_flags||[]).forEach(s=>{{
      if(s.date) (crisisMap[s.date]||(crisisMap[s.date]=[])).push(s.text.slice(0,140));
    }});
    if(Object.keys(crisisMap).length) {{
      sentDatasets.push({{
        label:label?`Crisis flag (${{label}})`:"Crisis flag",
        data:allDates.map(d=>crisisMap[d]?-0.97:null),
        _tips:crisisMap,
        backgroundColor:"rgba(239,68,68,.9)",borderColor:"rgba(239,68,68,1)",
        pointStyle:"triangle",pointRadius:9,pointHoverRadius:12,
        showLine:false,spanGaps:false
      }});
    }}
  }});

  requestAnimationFrame(function(){{
    destroyChart("sentChart");
    const ctx=document.getElementById("sentChart"); if(!ctx) return;
    charts["sentChart"]=new Chart(ctx,{{
      type:"line",
      data:{{labels:allDates,datasets:sentDatasets}},
      options:{{
        responsive:true,maintainAspectRatio:false,
        onClick:function(evt,elements){{
          if(!elements||!elements.length) return;
          const idx=elements[0].index;
          const label=allDates[idx];
          if(label) showMoodExplainer("sentChart",label);
        }},
        onHover:function(evt){{
          if(evt.native) evt.native.target.style.cursor='pointer';
        }},
        plugins:{{
          legend:{{position:"top"}},
          tooltip:{{
            mode:"index",
            callbacks:{{
              label:function(ctx2){{
                const ds=ctx2.dataset;
                if(ds._tips){{
                  const texts=ds._tips[allDates[ctx2.dataIndex]];
                  return texts ? ds.label+': "'+texts[0]+'"' : null;
                }}
                if(ctx2.parsed.y===null||ctx2.parsed.y===undefined) return null;
                return ctx2.dataset.label+': '+ctx2.parsed.y.toFixed(2);
              }}
            }}
          }}
        }},
        scales:{{
          x:{{ticks:{{maxTicksLimit:12,maxRotation:45}}}},
          y:{{min:-1,max:1,ticks:{{stepSize:0.5}}}}
        }}
      }}
    }});
  }});

  // ── Volume chart ─────────────────────────────────────────────────────────
  const volDatasets=[];
  // Merge all weeks across chats
  const allWeeks=[...new Set(chats.flatMap(c=>(c.weekly_volume&&c.weekly_volume.weeks)||[]))].sort();
  const barColors=["rgba(37,211,102,.7)","rgba(26,115,232,.7)","rgba(233,30,99,.7)","rgba(255,152,0,.7)"];
  const barColorsContact=["rgba(37,211,102,.35)","rgba(26,115,232,.35)","rgba(233,30,99,.35)","rgba(255,152,0,.35)"];
  chats.forEach((chat,i)=>{{
    if(!chat.weekly_volume||!chat.weekly_volume.weeks) return;
    const wkMap={{}};
    chat.weekly_volume.weeks.forEach((w,j)=>{{
      wkMap[w]={{u:chat.weekly_volume.user[j],c:chat.weekly_volume.contact[j]}};
    }});
    const label=chats.length>1?chat.name:"";
    volDatasets.push({{
      label:label?`You (${{label}})`:"You",
      data:allWeeks.map(w=>wkMap[w]?wkMap[w].u:0),
      backgroundColor:barColors[i%barColors.length],borderRadius:3,borderWidth:0
    }});
    volDatasets.push({{
      label:label?chat.name:chat.name,
      data:allWeeks.map(w=>wkMap[w]?wkMap[w].c:0),
      backgroundColor:barColorsContact[i%barColorsContact.length],borderRadius:3,borderWidth:0
    }});
  }});
  // Thin x-axis labels for volume — show every nth week
  const volLabels=allWeeks.map((w,i)=>{{
    const step=allWeeks.length>104?8:allWeeks.length>52?4:allWeeks.length>26?2:1;
    return i%step===0?w:"";
  }});
  requestAnimationFrame(()=>makeBarChart("volChart",volLabels,volDatasets));

  // ── Intensity chart ──────────────────────────────────────────────────────
  // "How emotionally charged were messages, regardless of positive/negative?"
  var intCard=document.createElement('div'); intCard.className='card';
  intCard.innerHTML=`<h2>Emotional intensity</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:14px">
      How strongly felt messages are each week — regardless of whether positive or negative.
      0 = completely neutral tone. 1 = maximum intensity. Useful for spotting periods of
      high emotional charge even when sentiment score is mixed.
    </p>
    <div class="chart-wrap-sm"><canvas id="intChart"></canvas></div>`;
  el.appendChild(intCard);
  const intDatasets=[];
  chats.forEach((chat,i)=>{{
    const uInt=absWeekly(chat.daily_sentiment.user||{{}});
    const cInt=absWeekly(chat.daily_sentiment.contact||{{}});
    const allIntWeeks=[...new Set([...uInt.weeks,...cInt.weeks])].sort();
    const uMap={{}}, cMap={{}};
    uInt.weeks.forEach((w,j)=>uMap[w]=uInt.vals[j]);
    cInt.weeks.forEach((w,j)=>cMap[w]=cInt.vals[j]);
    const lbl=chats.length>1?chat.name:"";
    intDatasets.push({{
      label:lbl?`You (${{lbl}})`:"You",
      data:allIntWeeks.map(w=>uMap[w]??null),
      borderColor:colors[i%colors.length],backgroundColor:colors[i%colors.length]+"22",
      borderWidth:2,pointRadius:1,spanGaps:true,tension:.3
    }});
    intDatasets.push({{
      label:lbl?chat.name:chat.name,
      data:allIntWeeks.map(w=>cMap[w]??null),
      borderColor:colorsDash[i%colorsDash.length],backgroundColor:"transparent",
      borderWidth:2,borderDash:[4,4],pointRadius:1,spanGaps:true,tension:.3
    }});
    // Re-use allIntWeeks for x-axis (use first chat's since all should be similar)
    if(i===0) requestAnimationFrame(function(){{
      destroyChart("intChart");
      var ctx=document.getElementById("intChart"); if(!ctx) return;
      charts["intChart"]=new Chart(ctx,{{type:"line",data:{{labels:allIntWeeks,datasets:intDatasets}},
        options:{{responsive:true,maintainAspectRatio:false,
          plugins:{{legend:{{position:"top"}}}},
          scales:{{x:{{ticks:{{maxTicksLimit:14,maxRotation:45}}}},y:{{min:0,max:1,ticks:{{stepSize:0.25}}}}}}}}}});
    }});
  }});

  // ── Divergence chart ─────────────────────────────────────────────────────
  // "On the same day, how differently are you and the contact feeling?"
  var divCard=document.createElement('div'); divCard.className='card';
  divCard.innerHTML=`<h2>Sentiment gap — you vs them</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:14px">
      Your mood score minus theirs, on days where both sent messages.
      <b>Above 0</b>: you were more positive than them.
      <b>Below 0</b>: they were more positive than you.
      Large persistent gaps can signal disconnection or one-sided emotional labour.
    </p>
    <div class="chart-wrap-sm"><canvas id="divChart"></canvas></div>`;
  el.appendChild(divCard);
  const divDatasets=[];
  const divColors=["#7c3aed","#0891b2","#be185d","#b45309"];
  chats.forEach((chat,i)=>{{
    const diff=dailyDiff(chat.daily_sentiment.user||{{}},chat.daily_sentiment.contact||{{}});
    const avgDiff=rollingAvg(diff,7);
    const diffDates=Object.keys(avgDiff).sort();
    const lbl=chats.length>1?chat.name:"";
    divDatasets.push({{
      label:lbl||"Divergence",
      data:diffDates.map(d=>avgDiff[d]??null),
      borderColor:divColors[i%divColors.length],
      backgroundColor:function(ctx2){{
        const v=ctx2.raw; return v==null?"transparent":v>=0?"rgba(16,185,129,.15)":"rgba(239,68,68,.15)";
      }},
      borderWidth:2,pointRadius:1,spanGaps:true,tension:.3,fill:true
    }});
    if(i===0) requestAnimationFrame(function(){{
      destroyChart("divChart");
      var ctx=document.getElementById("divChart"); if(!ctx) return;
      charts["divChart"]=new Chart(ctx,{{type:"line",data:{{labels:diffDates,datasets:divDatasets}},
        options:{{responsive:true,maintainAspectRatio:false,
          plugins:{{legend:{{position:"top"}}}},
          scales:{{
            x:{{ticks:{{maxTicksLimit:14,maxRotation:45}}}},
            y:{{min:-1,max:1,ticks:{{stepSize:0.5}},
              grid:{{color:function(ctx3){{return ctx3.tick.value===0?"#94a3b8":"#e2e8f0";}}}}}}}}}}}});
    }});
  }});

  // ── Emotion over time ────────────────────────────────────────────────────
  renderEmotionOverTime(el, chats);

  // ── Emotion breakdown ─────────────────────────────────────────────────────
  chats.forEach(function(chat){{
    const emoSum=chat.emotion_summary||{{}};
    if(!emoSum.user&&!emoSum.contact) return;
    var emoCard=document.createElement('div'); emoCard.className='card';
    const canvasId="emoChart_"+chat.id;
    emoCard.innerHTML=`<h2>Emotion breakdown${{chats.length>1?" — "+esc(chat.name):""}}</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:14px">
        Average Ekman emotion distribution across all messages.
        Each bar shows the percentage of messages where that emotion was dominant.
      </p>
      <div class="chart-wrap-sm"><canvas id="${{canvasId}}"></canvas></div>`;
    el.appendChild(emoCard);
    requestAnimationFrame(function(){{
      destroyChart(canvasId);
      const ctx=document.getElementById(canvasId); if(!ctx) return;
      const userEmo=emoSum.user||{{}};
      const contEmo=emoSum.contact||{{}};
      charts[canvasId]=new Chart(ctx,{{
        type:"bar",
        data:{{
          labels:["You",esc(chat.name)],
          datasets:EMOTION_LABELS.map(function(e){{
            return {{
              label:e,
              data:[userEmo[e]||0,contEmo[e]||0],
              backgroundColor:(EMOTION_COLORS[e]||"#9ca3af")+"cc"
            }};
          }})
        }},
        options:{{
          indexAxis:"y",responsive:true,maintainAspectRatio:false,
          plugins:{{legend:{{position:"bottom",labels:{{boxWidth:12,font:{{size:11}}}}}}}},
          scales:{{
            x:{{stacked:true,max:100,ticks:{{callback:function(v){{return v+"%";}}}}}},
            y:{{stacked:true}}
          }}
        }}
      }});
    }});
  }});
}}

// ── Emotion over Time ─────────────────────────────────────────────────────
function renderEmotionOverTime(el, chats) {{
  const ECOL = EMOTION_COLORS;
  const EMO_ORDER = ["joy","surprise","neutral","fear","disgust","sadness","anger"];
  chats.forEach(function(chat) {{
    const emo = chat.emotion_monthly;
    if (!emo || !emo.months || emo.months.length < 2) return;
    // reorder series so positive emotions are at base of stack
    const orderedSeries = {{}};
    EMO_ORDER.forEach(e => {{ if (emo.series[e]) orderedSeries[e] = emo.series[e]; }});
    const canvasId = "emoTimeChart_" + chat.id;
    const seriesJson = JSON.stringify(orderedSeries).replace(/"/g,'&quot;');
    const colormapJson = JSON.stringify(ECOL).replace(/"/g,'&quot;');
    const monthsJson = JSON.stringify(emo.months).replace(/"/g,'&quot;');
    const section = document.createElement("div");
    section.className = "card chart-section-wrap";
    section.dataset.series = JSON.stringify(orderedSeries);
    section.dataset.colormap = JSON.stringify(ECOL);
    section.dataset.months = JSON.stringify(emo.months);
    section.innerHTML = `
      <h2 style="margin-bottom:4px">Emotions Over Time</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:10px">Monthly breakdown of dominant emotion per message.</p>
      <div class="chart-toggle-bar">
        <button class="chart-toggle active" onclick="switchChartMode('${{canvasId}}','trend',this)">Trend</button>
        <button class="chart-toggle" onclick="switchChartMode('${{canvasId}}','stack',this)">100% Stack</button>
      </div>
      <div style="position:relative;height:240px"><canvas id="${{canvasId}}"></canvas></div>`;
    el.appendChild(section);
    requestAnimationFrame(() => drawToggleChart(canvasId, orderedSeries, ECOL, emo.months, "trend"));
  }});
}}

// ── Initiation balance ─────────────────────────────────────────────────────
function renderInitiation(el, chats) {{
  let html=`<div class="card"><h2>Initiation Balance</h2>`;
  chats.forEach(chat=>{{
    const init=chat.initiation_balance;
    html+=`<div style="margin-bottom:20px">
      ${{chats.length>1?`<h3>${{esc(chat.name)}}</h3>`:""}}
      <div class="bar-label"><span>You: ${{init.user_pct}}% (${{init.user_count}})</span><span>${{esc(chat.name)}}: ${{init.contact_pct}}% (${{init.contact_count}})</span></div>
      <div class="bar"><div class="bar-user" style="width:${{init.user_pct}}%"></div><div class="bar-contact" style="width:${{init.contact_pct}}%"></div></div>
      <div style="font-size:12px;color:var(--muted)">Trend: ${{init.trend}}</div>
    </div>`;
  }});
  html+=`</div>`;
  el.innerHTML=html;
}}

// ── Crisis ─────────────────────────────────────────────────────────────────
function renderCrisis(el, chat) {{
  renderCrisisContent(el,[chat]);
}}
function renderAllCrisis(el) {{
  renderCrisisContent(el,CHATS);
}}
function renderCrisisContent(el, chats) {{
  let html=`
    <div class="card">
      <h2>Crisis Moments</h2>
      <p style="font-size:13px;color:var(--muted);margin-bottom:16px">
        These flags are for personal review only. They are not diagnostic tools.
      </p>
      <button class="reveal-btn" onclick="document.getElementById('crisisContent').style.display='block';this.style.display='none'">Click to reveal flagged messages</button>
      <div id="crisisContent" class="crisis-content">`;
  let anyFlag=false;
  chats.forEach(chat=>{{
    if(!chat.crisis_flags.length) return;
    anyFlag=true;
    if(chats.length>1) html+=`<h3 style="margin-bottom:8px">${{esc(chat.name)}}</h3>`;
    chat.crisis_flags.forEach((f,i)=>{{
      const assessed=CRISIS_ASSESSED[chat.name]?.[i];
      const literalBadge=assessed&&assessed.is_literal_risk===true
        ?`<span style="background:#fee2e2;color:#991b1b;font-size:11px;padding:2px 6px;border-radius:3px;margin-left:6px;font-weight:600">confirmed risk</span>`:"";
      const dimLabels={{cognitive:"Cognitive",behavioral:"Behavioral",emotional:"Emotional",unknown:"Unknown"}};
      const levelColors={{1:"#d97706",2:"#dc2626",3:"#7c1d1d"}};
      const levelBg={{1:"#fef3c7",2:"#fee2e2",3:"#fce7e7"}};
      const levelLabels={{1:"Moderate Risk",2:"Elevated Risk",3:"Critical Risk"}};
      const dim=f.crisis_dimension||"unknown";
      const lvl=f.crisis_level||2;
      const src=f.crisis_source||"tier1_regex";
      const srcLabel=src==="tier2_rf"?" · ML":"";
      const dimBadge=`<span style="background:#e0e7ff;color:#3730a3;font-size:11px;padding:2px 6px;border-radius:3px;margin-left:6px;font-weight:600">${{dimLabels[dim]||dim}}</span>`;
      const levelBadge=`<span style="background:${{levelBg[lvl]}};color:${{levelColors[lvl]}};font-size:11px;padding:2px 6px;border-radius:3px;margin-left:4px;font-weight:600">${{levelLabels[lvl]||"Risk"}}${{srcLabel}}</span>`;
      html+=`<div style="border:1px solid var(--danger);border-radius:var(--radius);padding:12px;margin-bottom:10px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${{fmt(f.date)}}${{f.sender?` · <b>${{esc(f.sender)}}</b>`:""}}${{dimBadge}}${{levelBadge}}${{literalBadge}}</div>
        <div style="font-size:13px">${{esc(f.text)}}</div>
        ${{assessed?`<div style="margin-top:8px;font-size:12px;background:var(--bg);padding:8px;border-radius:4px">
          <b>AI assessment:</b> ${{esc(assessed.assessment)}} (${{esc(assessed.confidence)}} confidence)
        </div>`:""}}</div>`;
    }});
  }});
  if(!anyFlag) html+=`<p class="no-data">No crisis flags detected.</p>`;
  html+=`</div>
      <div class="crisis-safe">
        <b>If you are concerned about immediate safety:</b><br>
        Call <b>999</b> (emergency) or <b>Samaritans on 116 123</b> (free, 24/7, UK)
      </div>
    </div>`;
  el.innerHTML=html;
}}

// ── Support given ─────────────────────────────────────────────────────────
// ── Coping Dynamics ───────────────────────────────────────────────────────
function renderCoping(el, chats) {{
  const SUB_LABELS = {{
    self_care:"Self-Care",seeking_help:"Seeking Help",prayer_meditation:"Prayer/Meditation",
    adaptive_humor:"Adaptive Humor",hopeless:"Hopeless/Passive",
    avoidance:"Avoidance/Wishful",conspiracy_paranoia:"Conspiracy/Paranoia"
  }};
  const POS_THEMES=["self_care","seeking_help","prayer_meditation","adaptive_humor"];
  let html=`<div class="card"><h2>Coping Dynamics</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:16px">
      DRIVE framework — messages are classified into positive coping strategies
      (self-care, seeking help, adaptive humour) and negative ones (avoidance, hopeless thinking).
    </p>`;
  chats.forEach(function(chat){{
    const cs=chat.coping_summary||{{}};
    const ai=COPING_AI[chat.name]||{{}};
    const pos=cs.positive_count||0;
    const neg=cs.negative_count||0;
    const total=pos+neg||1;
    const canvasId="copingChart_"+chat.id;
    html+=`${{chats.length>1?`<h3 style="margin-bottom:8px">${{esc(chat.name)}}</h3>`:""}}
      <div style="display:flex;gap:24px;align-items:center;margin-bottom:16px">
        <div style="width:140px;height:140px;flex-shrink:0"><canvas id="${{canvasId}}"></canvas></div>
        <div style="flex:1">
          <div style="font-size:13px;margin-bottom:4px">
            <span style="color:#22c55e;font-weight:600">Positive: ${{pos}}</span>
            &nbsp;(<span>${{cs.positive_pct||0}}%</span>)
          </div>
          <div style="font-size:13px;margin-bottom:12px">
            <span style="color:#ef4444;font-weight:600">Negative: ${{neg}}</span>
            &nbsp;(<span>${{cs.negative_pct||0}}%</span>)
          </div>
          ${{ai.assessment?`<div style="font-size:13px;margin-bottom:6px">${{esc(ai.assessment)}}</div>`:"" }}
          ${{ai.recommendations?`<div style="font-size:12px;color:var(--muted);font-style:italic">${{esc(ai.recommendations)}}</div>`:"" }}
        </div>
      </div>`;
    // Sub-theme table
    const by=cs.by_subtheme||{{}};
    const subKeys=Object.keys(by).sort((a,b)=>by[b]-by[a]);
    if(subKeys.length){{
      html+=`<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:12px">
        <thead><tr>
          <th style="text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)">Sub-theme</th>
          <th style="text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)">Type</th>
          <th style="text-align:right;padding:4px 8px;border-bottom:1px solid var(--border)">Count</th>
        </tr></thead><tbody>`;
      subKeys.forEach(function(sub){{
        const isPos=POS_THEMES.includes(sub);
        const exs=(cs.examples||{{}})[sub]||[];
        html+=`<tr>
          <td style="padding:4px 8px;border-bottom:1px solid var(--border)">${{esc(SUB_LABELS[sub]||sub)}}</td>
          <td style="padding:4px 8px;border-bottom:1px solid var(--border);color:${{isPos?"#22c55e":"#ef4444"}}">${{isPos?"Positive":"Negative"}}</td>
          <td style="text-align:right;padding:4px 8px;border-bottom:1px solid var(--border)">${{by[sub]}}</td>
        </tr>`;
        if(exs.length){{
          html+=`<tr><td colspan="3" style="padding:0 8px 8px">`;
          exs.forEach(function(e){{
            html+=`<div style="font-size:11px;color:var(--muted);padding:2px 0;border-left:2px solid var(--border);padding-left:6px;margin-top:2px">
              <b>${{esc(e.sender)}}:</b> "${{esc(e.text.slice(0,140))}}"</div>`;
          }});
          html+=`</td></tr>`;
        }}
      }});
      html+=`</tbody></table>`;
    }} else {{
      html+=`<p class="no-data">No coping signals detected in this chat.</p>`;
    }}
  }});
  html+=`</div>`;
  el.innerHTML=html;
  // Render donut charts after DOM is set
  chats.forEach(function(chat){{
    const cs=chat.coping_summary||{{}};
    const pos=cs.positive_count||0;
    const neg=cs.negative_count||0;
    if(pos+neg===0) return;
    const canvasId="copingChart_"+chat.id;
    requestAnimationFrame(function(){{
      destroyChart(canvasId);
      const ctx=document.getElementById(canvasId); if(!ctx) return;
      charts[canvasId]=new Chart(ctx,{{
        type:"doughnut",
        data:{{
          labels:["Positive","Negative"],
          datasets:[{{data:[pos,neg],backgroundColor:["#22c55e","#ef4444"],borderWidth:1}}]
        }},
        options:{{
          responsive:true,maintainAspectRatio:true,
          plugins:{{legend:{{position:"bottom",labels:{{boxWidth:12,font:{{size:11}}}}}}}}
        }}
      }});
    }});
  }});
}}

function renderSupport(el, chats) {{
  let html=`<div class="card"><h2>Support Given (financial & practical)</h2>`;
  let anyItem=false;
  chats.forEach(chat=>{{
    if(!chat.financial_requests.length) return;
    anyItem=true;
    if(chats.length>1) html+=`<h3 style="margin-bottom:8px">${{esc(chat.name)}}</h3>`;
    chat.financial_requests.forEach(r=>{{
      html+=`<div style="padding:8px 0;border-bottom:1px solid var(--border);font-size:13px">
        <span style="color:var(--muted);font-size:11px">${{fmt(r.date)}} · ${{esc(r.sender)}}</span><br>
        ${{esc(r.text)}}
      </div>`;
    }});
  }});
  if(!anyItem) html+=`<p class="no-data">No financial or practical help requests detected.</p>`;
  html+=`</div>`;
  el.innerHTML=html;
}}

// ── People ─────────────────────────────────────────────────────────────────
function renderPeople(el) {{
  let names=Object.keys(PEOPLE);
  if(currentChat&&currentChat!=="all") {{
    const chat=getChatById(currentChat);
    if(chat) names=names.filter(n=>PEOPLE[n].chats&&PEOPLE[n].chats.includes(chat.name));
  }}
  if(!names.length) {{ el.innerHTML=`<p class="no-data">No recurring names detected (3+ mentions required).</p>`; return; }}
  let html=`<div class="card"><h2>People & Sentiment</h2>
    <p style="font-size:13px;color:var(--muted);margin-bottom:16px">Cards are editable — your changes are saved in your browser.</p>`;
  names.forEach(name=>{{
    const p=PEOPLE[name];
    const ai=PEOPLE_AI[name]||{{}};
    const saved=loadPersonEdit(name);
    const summary=saved||ai.summary||"";
    const label=ai.sentiment_label||"";
    const warn=ai.warning_flag||"";
    const sd=p.sentiment_dist||{{}};
    const hasAbsa=typeof p.avg_valence==="number";
    let valenceHtml="";
    if(hasAbsa){{
      const av=p.avg_valence;
      const pct=Math.round(((av+1)/2)*100);
      const col=av>0.1?"#22c55e":av<-0.1?"#ef4444":"#9ca3af";
      valenceHtml=`<div style="margin-bottom:8px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:3px">
          ABSA valence: <b style="color:${{col}}">${{av>=0?"+":""}}${{av.toFixed(2)}}</b>
          <span style="font-size:10px;color:var(--muted)"> (−1.0 negative → +1.0 positive)</span>
        </div>
        <div style="position:relative;height:8px;border-radius:4px;background:linear-gradient(to right,#ef4444,#9ca3af,#22c55e);overflow:visible">
          <div style="position:absolute;top:-3px;left:calc(${{pct}}% - 2px);width:4px;height:14px;background:#111;border-radius:2px"></div>
        </div>
      </div>`;
    }} else {{
      valenceHtml=`<div class="sentiment-bar">
        <div class="sb-pos" style="width:${{sd.positive||0}}%"></div>
        <div class="sb-neu" style="width:${{sd.neutral||0}}%"></div>
        <div class="sb-neg" style="width:${{sd.negative||0}}%"></div>
      </div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:8px">+${{sd.positive||0}}% / =${{sd.neutral||0}}% / -${{sd.negative||0}}%</div>`;
    }}
    const absa_excerpts=p.aspect_excerpts||[];
    html+=`<div class="person-card">
      <div class="name">${{esc(name)}} <span style="font-size:12px;color:var(--muted);font-weight:400">×${{p.count}} mentions · ${{p.chats.join(", ")}}</span></div>
      ${{label?`<div class="label">${{esc(label)}}</div>`:""}}
      ${{valenceHtml}}
      ${{warn?`<div style="background:#fef3c7;border-radius:4px;padding:8px;font-size:12px;margin-bottom:8px">⚠ ${{esc(warn)}}</div>`:""}}
      ${{summary?`<div class="summary">${{esc(summary)}}</div>`:""}}
      <textarea class="note" placeholder="Add your own notes about ${{name}}…" onchange="savePersonEdit('${{name}}',this.value)">${{esc(saved)}}</textarea>
      ${{absa_excerpts.length?`<details style="margin-top:8px"><summary style="font-size:12px;color:var(--muted);cursor:pointer">Show ABSA reasoning excerpts (${{absa_excerpts.length}})</summary><div style="margin-top:8px">
        ${{absa_excerpts.map(e=>{{
          const vc=e.valence>0.1?"#22c55e":e.valence<-0.1?"#ef4444":"#9ca3af";
          return `<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px">
            <span style="color:${{vc}};font-weight:600">${{e.valence>=0?"+":""}}${{e.valence.toFixed(2)}}</span>
            — "${{esc(e.text.slice(0,180))}}"
            ${{e.reasoning?`<div style="font-size:11px;color:var(--muted);font-style:italic;margin-top:2px">${{esc(e.reasoning)}}</div>`:""}}
          </div>`;
        }}).join("")}}
      </div></details>`
      :p.excerpts&&p.excerpts.length?`<details style="margin-top:8px"><summary style="font-size:12px;color:var(--muted);cursor:pointer">Show message excerpts</summary><div style="margin-top:8px">
        ${{p.excerpts.map(e=>`<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px">"${{esc(e.slice(0,200))}}"</div>`).join("")}}
      </div></details>`:""
      }}</div>`;
  }});
  html+=`</div>`;
  el.innerHTML=html;
}}

// ── LDA topic isolation helpers ──────────────────────────────────────────
function toggleLdaTopic(cardEl, color) {{
  const canvasId=cardEl.dataset.canvas;
  const label=cardEl.dataset.label;
  const chart=charts[canvasId];
  if(!chart) return;
  const cards=document.querySelectorAll(`[data-canvas="${{canvasId}}"]`);
  const alreadyIsolated=Array.from(cards).some(c=>c.dataset.isolated==="1");
  if(alreadyIsolated && cardEl.dataset.isolated==="1") {{
    // click isolated card again → restore all
    chart.data.datasets.forEach(d=>{{d.hidden=false;}});
    cards.forEach(c=>{{c.dataset.isolated="";c.style.opacity="1";c.style.borderColor=c.dataset.color+"22";}});
  }} else if(alreadyIsolated) {{
    // click a different card → switch isolation
    chart.data.datasets.forEach(d=>{{d.hidden=(d.label!==label);}});
    cards.forEach(c=>{{
      const active=c.dataset.label===label;
      c.dataset.isolated=active?"1":"";
      c.style.opacity=active?"1":"0.35";
      c.style.borderColor=active?c.dataset.color:c.dataset.color+"22";
    }});
  }} else {{
    // no isolation yet → isolate this card
    chart.data.datasets.forEach(d=>{{d.hidden=(d.label!==label);}});
    cards.forEach(c=>{{
      const active=c.dataset.label===label;
      c.dataset.isolated=active?"1":"";
      c.style.opacity=active?"1":"0.35";
      c.style.borderColor=active?c.dataset.color:c.dataset.color+"22";
    }});
  }}
  chart.update();
}}
function ldaCardHoverOut(cardEl, color) {{
  if(!cardEl.dataset.isolated) cardEl.style.borderColor=color+"22";
}}

// ── Topics Over Time ──────────────────────────────────────────────────────
function renderTopics(el, chats) {{
  const TOPIC_COLORS=["#6366f1","#22c55e","#f97316","#3b82f6","#a855f7","#eab308","#ec4899","#14b8a6"];
  let html=`<div class="card"><h2>Topics Over Time</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:16px">
      LDA probabilistic topic discovery. Each month shows the distribution of dominant topics.
      Falls back to keyword topics when gensim is not installed.
    </p>`;
  let anyLda=false;
  chats.forEach(function(chat){{
    const lda=chat.lda_topics||{{}};
    const kw=chat.topics||{{}};
    if(chats.length>1) html+=`<h3 style="margin-bottom:8px">${{esc(chat.name)}}</h3>`;
    if(lda.months&&lda.months.length>1&&lda.series){{
      anyLda=true;
      const canvasId="ldaChart_"+chat.id;
      const ldaSeriesJson=JSON.stringify(lda.series).replace(/"/g,'&quot;');
      const ldaColorMap={{}};
      Object.keys(lda.series).forEach((l,i)=>{{ ldaColorMap[l]=TOPIC_COLORS[i%TOPIC_COLORS.length]; }});
      const ldaColorJson=JSON.stringify(ldaColorMap).replace(/"/g,'&quot;');
      const ldaMonthsJson=JSON.stringify(lda.months).replace(/"/g,'&quot;');
      html+=`<div class="chart-section-wrap" data-series="${{ldaSeriesJson}}" data-colormap="${{ldaColorJson}}" data-months="${{ldaMonthsJson}}" style="margin-bottom:24px">
        <div class="chart-toggle-bar">
          <button class="chart-toggle active" onclick="switchChartMode('${{canvasId}}','trend',this)">Trend</button>
          <button class="chart-toggle" onclick="switchChartMode('${{canvasId}}','stack',this)">100% Stack</button>
        </div>
        <div style="position:relative;height:200px"><canvas id="${{canvasId}}"></canvas></div>
        <p style="font-size:11px;color:var(--muted);margin:6px 0 8px;text-align:center">Click a topic card to isolate it · click again to restore all</p>
        <div style="margin-top:4px;display:flex;flex-wrap:wrap;gap:8px" id="lda-cards-${{canvasId}}">`;
      const labels=Object.keys(lda.series);
      labels.forEach(function(label,i){{
        const words=(lda.topic_words||{{}})[label]||[];
        const color=TOPIC_COLORS[i%TOPIC_COLORS.length];
        html+=`<div class="lda-topic-card" data-canvas="${{canvasId}}" data-label="${{esc(label)}}" data-color="${{color}}"
          style="background:var(--bg);border:2px solid ${{color}}22;border-radius:var(--radius);padding:8px 12px;font-size:12px;flex:1;min-width:160px;cursor:pointer;transition:border-color .15s,opacity .15s"
          onmouseenter="this.style.borderColor='${{color}}'" onmouseleave="ldaCardHoverOut(this,'${{color}}')"
          onclick="toggleLdaTopic(this,'${{color}}')">
          <div style="font-weight:600;color:${{color}};margin-bottom:4px">${{esc(label)}}</div>
          <div style="color:var(--muted)">${{words.slice(0,8).map(w=>`<span style="background:var(--surface);padding:1px 5px;border-radius:3px;margin:1px;display:inline-block">${{esc(w)}}</span>`).join("")}}</div>
        </div>`;
      }});
      html+=`</div></div>`;
    }} else if(kw.counts&&Object.keys(kw.counts).length){{
      // Fallback: keyword topic bar chart
      html+=`<p style="font-size:12px;color:var(--muted);margin-bottom:8px">Keyword topics (install gensim for LDA):</p>`;
      const counts=kw.counts;
      const sorted=Object.keys(counts).sort((a,b)=>counts[b]-counts[a]);
      const max=counts[sorted[0]]||1;
      sorted.forEach(function(topic){{
        const pct=Math.round(counts[topic]/max*100);
        const pretty=topic.replace(/_/g," ");
        html+=`<div style="margin-bottom:8px">
          <div style="font-size:13px;margin-bottom:3px">${{esc(pretty)}} <span style="color:var(--muted);font-size:11px">(${{counts[topic]}} messages)</span></div>
          <div style="height:6px;border-radius:3px;background:var(--border);overflow:hidden">
            <div style="height:100%;width:${{pct}}%;background:#6366f1;border-radius:3px"></div>
          </div>
        </div>`;
      }});
    }} else {{
      html+=`<p class="no-data">No topic data — not enough messages or gensim not installed.</p>`;
    }}
  }});
  html+=`</div>`;
  el.innerHTML=html;
  // Draw LDA charts after DOM update
  chats.forEach(function(chat){{
    const lda=chat.lda_topics||{{}};
    if(!lda.months||lda.months.length<=1||!lda.series) return;
    const canvasId="ldaChart_"+chat.id;
    const colorMap={{}};
    Object.keys(lda.series).forEach((l,i)=>{{ colorMap[l]=TOPIC_COLORS[i%TOPIC_COLORS.length]; }});
    requestAnimationFrame(()=>drawToggleChart(canvasId, lda.series, colorMap, lda.months, "trend"));
  }});
}}

// ── Relationship Network ──────────────────────────────────────────────────
function renderNetwork(el) {{
  const fig=TOPOLOGY_FIGURE;
  if(!fig||!fig.data||!fig.data.length){{
    el.innerHTML=`<div class="card"><h2>Relationship Network</h2>
      <p class="no-data">Network graph requires networkx (pip install networkx) and at least
      two people detected across chats.</p></div>`;
    return;
  }}
  el.innerHTML=`<div class="card">
    <h2>Relationship Network</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px">
      Entity–sentiment graph. Node size = degree centrality. Colour = average valence
      (green → positive, red → negative, grey → neutral).
      <span style="color:#22c55e;font-weight:600">■</span> positive &nbsp;
      <span style="color:#ef4444;font-weight:600">■</span> negative &nbsp;
      <span style="color:#9ca3af;font-weight:600">■</span> neutral &nbsp;
      <span style="font-size:11px">Larger node = appears in more chats/contexts.</span>
    </p>
    <div id="topoPlot" style="width:100%;height:520px"></div>
  </div>`;
  if(typeof Plotly==="undefined"){{
    document.getElementById("topoPlot").innerHTML=`<p class="no-data">Plotly.js failed to load. Check network connection.</p>`;
    return;
  }}
  requestAnimationFrame(function(){{
    const div=document.getElementById("topoPlot");
    if(!div) return;
    try {{
      Plotly.newPlot(div, fig.data, fig.layout||{{}}, {{
        responsive:true, displayModeBar:false, staticPlot:false
      }});
    }} catch(e) {{
      div.innerHTML=`<p class="no-data">Network render error: ${{esc(String(e))}}</p>`;
    }}
  }});
}}

// ── Framework ─────────────────────────────────────────────────────────────
function renderFramework(el, chat) {{
  const suggestions=chat.framework_suggestions||[];
  let html=`<div class="card"><h2>Communications Framework — ${{esc(chat.name)}}</h2>`;
  if(chat.framework_content) {{
    html+=`<pre style="white-space:pre-wrap;font-size:13px;line-height:1.7;margin-bottom:16px;padding:16px;background:var(--bg);border-radius:var(--radius)">${{esc(chat.framework_content)}}</pre>`;
  }}
  if(suggestions.length) {{
    html+=`<h3 style="margin-bottom:10px">AI-suggested adjustments</h3>`;
    suggestions.forEach(s=>{{
      html+=`<div class="fw-item">
        <div class="fw-original">${{esc(s.framework_item)}}</div>
        <div class="fw-suggestion">Suggested: ${{esc(s.suggested_adjustment)}}</div>
        <div class="fw-basis">Based on: ${{esc(s.data_basis)}}</div>
      </div>`;
    }});
  }}
  html+=`</div>`;
  el.innerHTML=html;
}}

// ── Narrative vs Record ────────────────────────────────────────────────────
function renderNarrative(el, chat) {{
  const pairs=chat.narrative_vs_record||[];
  const corr=CROSS_CHAT.correlations||[];
  const relevant=corr.filter(c=>c.chats&&c.chats.includes(chat.name));
  let html=`<div class="card"><h2>Narrative vs Record — ${{esc(chat.name)}}</h2>`;
  if(!pairs.length) {{
    html+=`<p class="no-data">No contrasts generated. Run with AI enabled for this section.</p>`;
  }} else {{
    pairs.forEach(p=>{{
      html+=`<div class="contrast-pair">
        <div class="claimed">${{esc(p.claimed)}}</div>
        <div class="record">${{esc(p.record_shows)}}</div>
      </div>`;
    }});
  }}
  if(relevant.length) {{
    html+=`<h3 style="margin-top:16px;margin-bottom:8px">Cross-chat correlations</h3>`;
    relevant.forEach(c=>{{
      html+=`<div class="cross-chat-card">
        <div style="font-weight:600">${{fmt(c.date)}} — ${{esc(c.type)}}</div>
        <div>${{esc(c.summary)}}</div>
      </div>`;
    }});
  }}
  html+=`</div>`;
  el.innerHTML=html;
}}

// ── Relationship Summary ───────────────────────────────────────────────────
function renderSummary(el, chat) {{
  const text = chat.relationship_summary || "";
  let html = `<div class="card"><h2>Relationship Summary — ${{esc(chat.name)}}</h2>`;
  if(!text) {{
    html += `<p class="no-data">No summary generated. Run with AI enabled to generate a relationship summary.</p>`;
  }} else {{
    // Render each paragraph
    const paras = text.split(/\n+/).map(p => p.trim()).filter(Boolean);
    paras.forEach(function(p) {{
      html += `<p style="font-size:14px;line-height:1.75;margin-bottom:14px;color:var(--text)">${{esc(p)}}</p>`;
    }});
  }}
  html += `</div>`;
  el.innerHTML = html;
}}

// ── Chat Q&A ──────────────────────────────────────────────────────────────
function renderAsk(el, chatId) {{
  el.innerHTML = `<div class="card" id="ask-card">
    <h2>Ask AI about this chat${{chatId==="all"?" (all conversations)":""}}</h2>
    <p style="font-size:13px;color:var(--muted);margin-bottom:16px">
      Ask anything about the messages — patterns, events, themes, or specific people mentioned.
    </p>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <input type="text" id="ask-input" placeholder="e.g. What topics come up when she is upset?" style="flex:1;padding:9px 12px;border:1.5px solid var(--border);border-radius:7px;font-size:14px;color:var(--text);background:var(--bg);outline:none" onkeydown="if(event.key==='Enter')askQuestion()">
      <button onclick="askQuestion()" style="padding:9px 18px;border-radius:7px;border:none;background:var(--contact);color:#fff;font-size:14px;font-weight:600;cursor:pointer">Ask</button>
    </div>
    <div id="ask-history" style="margin-top:16px"></div>
  </div>`;

  // Store chatId on the card so askQuestion can read it
  document.getElementById("ask-card").dataset.chatId = chatId;
}}

function askQuestion() {{
  const card = document.getElementById("ask-card");
  if(!card) return;
  const chatId = card.dataset.chatId || currentChat;
  const input = document.getElementById("ask-input");
  const question = (input ? input.value : "").trim();
  if(!question) return;
  if(input) input.value = "";

  const history = document.getElementById("ask-history");
  if(!history) return;

  // Append question bubble (right-aligned)
  const qEl = document.createElement("div");
  qEl.style.cssText = "display:flex;justify-content:flex-end;margin-bottom:8px";
  qEl.innerHTML = `<div style="max-width:70%;background:var(--contact);color:#fff;padding:10px 14px;border-radius:18px;border-bottom-right-radius:4px;font-size:14px;line-height:1.5">${{esc(question)}}</div>`;
  history.appendChild(qEl);

  // Spinner
  const aEl = document.createElement("div");
  aEl.style.cssText = "display:flex;justify-content:flex-start;margin-bottom:16px";
  aEl.innerHTML = `<div style="max-width:80%;background:var(--surface);border:1px solid var(--border);padding:10px 14px;border-radius:18px;border-bottom-left-radius:4px;font-size:14px;line-height:1.5;color:var(--muted)">Thinking&#8230;</div>`;
  history.appendChild(aEl);
  history.scrollTop = history.scrollHeight;

  if(typeof pywebview === 'undefined' || !pywebview.api || !pywebview.api.chat_qa) {{
    aEl.firstChild.style.color = "var(--danger)";
    aEl.firstChild.textContent = "Chat Q&A is only available in the desktop app.";
    return;
  }}

  pywebview.api.chat_qa(chatId, question).then(function(answer) {{
    const paras = (answer||"No response").split(/\n+/).map(p=>p.trim()).filter(Boolean);
    aEl.firstChild.style.color = "var(--text)";
    aEl.firstChild.innerHTML = paras.map(p=>`<p style="margin:0 0 6px">${{esc(p)}}</p>`).join("");
    history.scrollTop = history.scrollHeight;
  }}).catch(function(err) {{
    aEl.firstChild.style.color = "var(--danger)";
    aEl.firstChild.textContent = "Error: " + String(err);
  }});
}}

// ── Mood Explainer ────────────────────────────────────────────────────────
function showMoodExplainer(canvasId, dateStr) {{
  // Find or create the explainer card after the chart's parent card
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const parentCard = canvas.closest(".card");
  if(!parentCard) return;

  let explainer = document.getElementById("mood-explainer-card");
  if(!explainer) {{
    explainer = document.createElement("div");
    explainer.id = "mood-explainer-card";
    explainer.className = "card";
    explainer.style.cssText = "margin-top:0;border-top:none;border-top-left-radius:0;border-top-right-radius:0";
    parentCard.insertAdjacentElement("afterend", explainer);
  }}

  explainer.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-weight:600;font-size:14px">Explaining mood around ${{esc(dateStr)}} &#8230;</span>
    <a href="#" onclick="document.getElementById('mood-explainer-card').remove();return false" style="font-size:13px;color:var(--muted)">&#215; Dismiss</a>
  </div>
  <div id="mood-explainer-body" style="font-size:14px;color:var(--muted);line-height:1.6">&#8987; Asking AI&#8230;</div>`;

  if(typeof pywebview === 'undefined' || !pywebview.api || !pywebview.api.explain_period) {{
    document.getElementById("mood-explainer-body").innerHTML = `<span style="color:var(--danger)">Mood Explainer is only available in the desktop app.</span>`;
    return;
  }}

  pywebview.api.explain_period(currentChat, dateStr).then(function(answer) {{
    const body = document.getElementById("mood-explainer-body");
    if(!body) return;
    const paras = (answer||"No response").split(/\n+/).map(p=>p.trim()).filter(Boolean);
    body.style.color = "var(--text)";
    body.innerHTML = paras.map(p=>`<p style="margin:0 0 8px">${{esc(p)}}</p>`).join("");
  }}).catch(function(err) {{
    const body = document.getElementById("mood-explainer-body");
    if(body) {{ body.style.color="var(--danger)"; body.textContent="Error: "+String(err); }}
  }});
}}

// ── Load notes from pywebview disk storage (desktop app only) ─────────────
// In a browser, pywebview is undefined so this never fires.
// In the desktop app, this syncs the disk notes.json into localStorage
// before the first render, so persisted notes appear immediately.
window.addEventListener('pywebviewready', function() {{
  if(typeof pywebview === 'undefined' || !pywebview.api || !pywebview.api.load_notes) return;
  pywebview.api.load_notes().then(function(notes) {{
    if(!notes) return;
    Object.entries(notes).forEach(function(kv) {{ localStorage.setItem(kv[0], kv[1]); }});
    renderMain();
  }}).catch(function(){{}});
}});

// ── Init ──────────────────────────────────────────────────────────────────
(function init() {{
  if(!CHATS.length) {{
    document.getElementById("mainContent").innerHTML=`<p class="no-data">No chat data loaded.</p>`;
    return;
  }}
  buildSidebar();
  const defaultBtn=CHATS.length>1
    ? document.querySelector('[data-chat="all"]')
    : document.querySelector(".sb-btn");
  if(defaultBtn) selectChat(defaultBtn);
}})();
</script>
</body>
</html>"""

    Path(output_file).write_text(html, encoding="utf-8")
    print(f"[INFO] HTML written: {output_file} ({Path(output_file).stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="WhatsApp Chat Analyser — generates a self-contained HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyser.py
  python analyser.py --no-ai
  python analyser.py --local-only --output report.html
  python analyser.py --claude-crisis --chat "Michelle O'Neill"
  python analyser.py --log-tokens
""",
    )
    p.add_argument("--no-ai", action="store_true", help="Skip all AI calls (Python analysis only)")
    p.add_argument("--local-only", action="store_true", help="Use Ollama only; skip Claude even for crisis")
    p.add_argument("--claude-crisis", action="store_true", help="Enable Claude API for crisis assessment")
    p.add_argument("--chat", metavar="NAME", help="Process only the named chat contact")
    p.add_argument("--output", metavar="PATH", default=None, help="Override output file path")
    p.add_argument("--log-tokens", action="store_true", help="Log token usage per AI call")
    return p.parse_args()


def main():
    args = parse_args()

    output_file = args.output or OUTPUT_FILE
    nlp_engine = NLP_ENGINE
    custom_topics = CUSTOM_TOPICS

    # Filter chats if --chat flag given
    chats_config = CHATS
    if args.chat:
        chats_config = [c for c in CHATS if c["contact_name"].lower() == args.chat.lower()]
        if not chats_config:
            print(f"[ERROR] No chat found with contact_name '{args.chat}'", file=sys.stderr)
            sys.exit(1)

    # Determine AI flags
    use_ai = not args.no_ai
    use_claude_crisis = (args.claude_crisis or CRISIS_AI_ASSESSMENT) and not args.local_only
    api_key = ANTHROPIC_API_KEY

    # ── Parse all chats
    chats = [parse_chat(cfg) for cfg in chats_config]
    chats = [c for c in chats if c["messages"]]
    if not chats:
        print("[ERROR] No messages parsed. Check your file paths in CHATS config.", file=sys.stderr)
        sys.exit(1)

    # ── Per-chat analysis
    for chat in chats:
        run_per_chat_analysis(chat, nlp_engine, custom_topics)

    # ── Cross-chat analysis
    cross_chat = run_cross_chat_analysis(chats)

    # ── AI calls
    ai_results = {
        "people_cards": {},
        "narrative_vs_record": {},
        "framework_suggestions": {},
        "timeline_highlights": [],
        "cross_chat_links": [],
        "crisis_assessed": {},
        "coping_analysis": {},
        "relationship_summary": {},
    }

    if use_ai:
        base_url = OLLAMA_BASE_URL
        model = OLLAMA_MODEL

        # Call 1 — people cards
        print("[AI] Generating people cards…")
        ai_results["people_cards"] = ai_people_cards(cross_chat["merged_people"], base_url, model)

        # Call 2 — narrative vs record (per chat)
        for chat in chats:
            print(f"[AI] Narrative vs record for {chat['contact_name']}…")
            ai_results["narrative_vs_record"][chat["contact_name"]] = ai_narrative_vs_record(chat, cross_chat, base_url, model)

        # Call 3 — framework personalisation (per chat with framework)
        for chat in chats:
            if chat.get("framework_content"):
                print(f"[AI] Framework personalisation for {chat['contact_name']}…")
                ai_results["framework_suggestions"][chat["contact_name"]] = ai_framework_personalisation(chat, base_url, model)

        # Call 4 — timeline highlights
        print("[AI] Timeline highlights…")
        ai_results["timeline_highlights"] = ai_timeline_highlights(chats, base_url, model)

        # Call 5 — cross-chat links
        if len(chats) > 1:
            print("[AI] Cross-chat link analysis…")
            ai_results["cross_chat_links"] = ai_cross_chat_links(cross_chat, chats, base_url, model)

        # Call 6 — crisis assessment (Claude if configured, otherwise local Ollama)
        for chat in chats:
            if not chat["analytics"]["crisis_flags"]:
                continue
            if use_claude_crisis:
                print(f"[AI] Crisis assessment via Claude for {chat['contact_name']}…")
                assessed = ai_crisis_assessment_claude(chat["analytics"]["crisis_flags"], api_key)
            else:
                print(f"[AI] Crisis assessment via Ollama for {chat['contact_name']}…")
                assessed = ai_crisis_assessment_ollama(
                    chat["analytics"]["crisis_flags"], base_url, model
                )
            ai_results["crisis_assessed"][chat["contact_name"]] = {
                i: item for i, item in enumerate(assessed)
            }

        # Call 7 — DRIVE coping analysis (per chat)
        for chat in chats:
            print(f"[AI] Coping analysis for {chat['contact_name']}…")
            ai_results["coping_analysis"][chat["contact_name"]] = ai_coping_analysis(
                chat, base_url, model
            )

        # Call 8 — ABSA entity-valence mapping (per chat)
        absa_per_chat = []
        for chat in chats:
            print(f"[AI] ABSA entity-sentiment for {chat['contact_name']}…")
            absa = run_absa(chat, base_url, model)
            absa_per_chat.append(absa)
        merge_absa_into_people(cross_chat["merged_people"], absa_per_chat)

        # Call 9 — relationship summary (per chat)
        for chat in chats:
            print(f"[AI] Relationship summary for {chat['contact_name']}…")
            ai_results["relationship_summary"][chat["contact_name"]] = ai_relationship_summary(
                chat, base_url, model
            )

    # ── Token log
    if args.log_tokens and TOKEN_LOG:
        print("\n[TOKEN LOG]")
        for entry in TOKEN_LOG:
            print(f"  {entry['model']}: {entry['tokens']} tokens")
        total = sum(e["tokens"] for e in TOKEN_LOG)
        print(f"  Total: {total} tokens")

    # ── Generate HTML
    generate_html(chats, cross_chat, ai_results, output_file)
    print(f"\nDone. Open {output_file} in your browser.")


if __name__ == "__main__":
    main()
