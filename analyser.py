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
OLLAMA_MODEL = "gemma3"          # adjust to your installed model name
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
    r"^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s[-–]\s(.+?):\s(.+)$"
)

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
            sender = m.group(3).strip()
            text = m.group(4).strip()
            if not _is_noise(text):
                counts[sender] = counts.get(sender, 0) + 1
                if sum(counts.values()) >= sample_size:
                    break
    return [s for s, _ in sorted(counts.items(), key=lambda x: -x[1])]


def parse_chat(config: dict) -> dict:
    """Parse a single WhatsApp export file. Returns a chat object."""
    path = config["file"]
    contact = config["contact_name"]
    messages = []

    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"[WARN] Chat file not found: {path}", file=sys.stderr)
        return _empty_chat(config)

    current = None
    for line in raw.splitlines():
        m = MSG_RE.match(line)
        if m:
            if current and not _is_noise(current["text"]):
                messages.append(current)
            date_str, time_str, sender, text = m.groups()
            dt = _parse_dt(date_str, time_str)
            sender = sender.strip()
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
            # continuation line
            current["text"] += " " + line.strip()

    if current and not _is_noise(current["text"]):
        messages.append(current)

    # Load framework file if specified
    framework_content = None
    fw_path = config.get("framework")
    if fw_path:
        try:
            framework_content = Path(fw_path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            print(f"[WARN] Framework file not found: {fw_path}", file=sys.stderr)

    return {
        "config": config,
        "messages": messages,
        "framework_content": framework_content,
        "contact_name": contact,
        "contact_relationship": config.get("contact_relationship", ""),
        "tags": config.get("tags", []),
    }


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
# NLP — SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

_sentiment_pipeline = None


def _load_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    try:
        from transformers import pipeline
        print("[INFO] Loading sentiment model (transformers)…")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        print("[INFO] Sentiment model loaded.")
    except Exception as e:
        print(f"[WARN] Could not load transformers pipeline: {e}. Falling back to TextBlob.")
        _sentiment_pipeline = "textblob"
    return _sentiment_pipeline


def score_sentiment_transformers(text: str, pipeline_obj) -> dict:
    if len(text.strip()) < 3:
        return {"label": "neutral", "score": 0.0}
    try:
        result = pipeline_obj(text[:512])[0]
        label = result["label"].lower()  # POSITIVE / NEGATIVE
        score = result["score"]
        if label == "positive":
            return {"label": "positive", "score": round(score, 4)}
        else:
            return {"label": "negative", "score": round(-score, 4)}
    except Exception:
        return {"label": "neutral", "score": 0.0}


def score_sentiment_textblob(text: str) -> dict:
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.05:
            label = "positive"
        elif polarity < -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(polarity, 4)}
    except Exception:
        return {"label": "neutral", "score": 0.0}


def add_sentiment_scores(chat: dict, engine: str = "transformers") -> None:
    """Attach sentiment scores to every message in the chat."""
    pipe = None
    if engine == "transformers":
        pipe = _load_sentiment_pipeline()

    for msg in chat["messages"]:
        if pipe and pipe != "textblob":
            s = score_sentiment_transformers(msg["text"], pipe)
        else:
            s = score_sentiment_textblob(msg["text"])
        msg["sentiment"] = s

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
]

CRISIS_KW = [
    r"\b(don['']t want to (be here|live|exist|carry on)|better off (without me|dead)|ending it|not worth living|want to die|kill myself|suicide)\b",
    r"\b(took.{0,20}(pills|tablets)|hurt myself|self.harm)\b",
]

FINANCIAL_KW = [
    r"\b(borrow|lend|loan|send|transfer|advance|owe|pay you back|need.{0,15}money|short.{0,10}this month)\b",
    r"\b(can you help.{0,20}(financially|with money|pay|afford))\b",
    r"\b(bills|rent|electricity|gas|mortgage).{0,20}(can['']t|struggling|behind|overdue)\b",
]


def _any_pattern(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def classify_intent(msg: dict) -> list:
    """Return a list of intent tags for a message."""
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
    if _any_pattern(t, CRISIS_KW):
        intents.append("CRISIS_FLAG")
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


def extract_person_mentions(chat: dict, min_count: int = 3) -> dict:
    """
    Extract names mentioned ≥ min_count times.
    Heuristic: capitalised words that are not the known sender names or common words.
    """
    known = {PRIMARY_USER_NAME.lower(), chat["contact_name"].lower()}
    # Add first names only too
    known.update(n.split()[0].lower() for n in known if n)

    STOP = {"i", "i'm", "i've", "i'll", "i'd", "me", "my", "we", "our", "she", "he",
            "her", "his", "them", "they", "it", "that", "this", "what", "when",
            "where", "how", "who", "why", "ok", "okay", "yes", "no", "hi", "hey",
            "oh", "so", "just", "got", "get", "don", "doesn", "didn", "isn", "wasn",
            "aren", "haven", "hadn", "wouldn", "couldn", "shouldn", "let", "going",
            "think", "know", "really", "well", "good", "great", "fine",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"}

    name_counter = Counter()
    name_messages = defaultdict(list)

    for msg in chat["messages"]:
        words = re.findall(r"\b[A-Z][a-z]{2,}\b", msg["text"])
        for w in words:
            wl = w.lower()
            if wl not in known and wl not in STOP:
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


def run_per_chat_analysis(chat: dict, engine: str, custom_topics: list) -> None:
    """Run all per-chat Python analysis, attaching results to the chat dict."""
    print(f"[INFO] Analysing chat: {chat['contact_name']}")
    add_sentiment_scores(chat, engine)
    add_intents(chat)
    label_exchanges(chat)

    # Re-run intent after exchange labelling (exchange start intents added in label_exchanges)
    chat["analytics"] = {
        "visit_tracker": compute_visit_tracker(chat),
        "initiation_balance": compute_initiation_balance(chat),
        "response_times": compute_response_times(chat),
        "daily_sentiment": compute_daily_sentiment(chat),
        "weekly_volume": compute_weekly_volume(chat),
        "topics": compute_topics(chat, custom_topics),
        "people": extract_person_mentions(chat),
        "distress_signals": [
            {"date": str(m["date"]), "text": m["text"][:300], "sender": m["sender"]}
            for m in chat["messages"] if "DISTRESS_SIGNAL" in m.get("intents", [])
        ],
        "crisis_flags": [
            {"date": str(m["date"]), "text": m["text"][:300]}
            for m in chat["messages"] if "CRISIS_FLAG" in m.get("intents", [])
        ],
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


def ai_crisis_assessment_claude(crisis_flags: list, api_key: str) -> list:
    """Call 6 — Claude API crisis assessment (optional)."""
    if not crisis_flags or not api_key:
        return []
    try:
        import anthropic
    except ImportError:
        print("[WARN] anthropic package not installed. Skipping crisis assessment.", file=sys.stderr)
        return []
    client = anthropic.Anthropic(api_key=api_key)
    excerpts = [{"id": i, "text": f["text"]} for i, f in enumerate(crisis_flags)]
    prompt = (
        "These messages have been flagged as potentially containing crisis language. "
        "For each, assess whether this appears to be genuine distress, rhetorical/emotional expression, "
        "or unclear. Return as JSON array: [{\"excerpt_id\": N, \"assessment\": \"...\", \"confidence\": \"high|medium|low\"}]. "
        "Do not include method or means information.\n\n"
        f"Excerpts:\n{json.dumps(excerpts)}"
    )
    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return _parse_json_response(raw, [])
    except Exception as e:
        print(f"[WARN] Claude API call failed: {e}", file=sys.stderr)
        return []

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
            "distress_signals": a["distress_signals"],
            "crisis_flags": a["crisis_flags"],
            "financial_requests": a["financial_requests"],
            "narrative_vs_record": ai_results.get("narrative_vs_record", {}).get(chat["contact_name"], []),
            "framework_suggestions": ai_results.get("framework_suggestions", {}).get(chat["contact_name"], []),
            "message_count": len(chat["messages"]),
            "messages_for_timeline": [
                {
                    "date": str(m["date"]),
                    "sender": m["sender"],
                    "is_user": m["is_user"],
                    "text": m["text"][:300],
                    "sentiment": m.get("sentiment", {}).get("label", "neutral"),
                    "intents": [i for i in m.get("intents", []) if i not in ("INITIATED_BY_USER", "INITIATED_BY_CONTACT")],
                }
                for m in chat["messages"]
            ],
        }
        chat_data.append(cd)

    people_data = cross_chat.get("merged_people", {})
    people_ai = ai_results.get("people_cards", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhatsApp Analysis — {_escape_html(PRIMARY_USER_NAME)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
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
.chat-selector{{background:var(--surface);border-bottom:1px solid var(--border);padding:8px 24px;display:flex;gap:8px;flex-wrap:wrap}}
.chat-btn{{padding:6px 14px;border-radius:20px;border:1px solid var(--border);background:var(--bg);cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}}
.chat-btn.active{{background:var(--contact);color:#fff;border-color:var(--contact)}}
.tabs{{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex;gap:0;overflow-x:auto}}
.tab-btn{{padding:12px 18px;border:none;background:transparent;cursor:pointer;font-size:14px;font-weight:500;color:var(--muted);border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s}}
.tab-btn.active{{color:var(--contact);border-bottom-color:var(--contact)}}
.main{{max-width:1200px;margin:0 auto;padding:24px}}
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
@media(max-width:600px){{
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

<div class="chat-selector" id="chatSelector">
  {"".join(f'<button class="chat-btn" data-chat="{re.sub(chr(92) + "W+", "_", c["contact_name"])}" onclick="selectChat(this)">{_escape_html(c["contact_name"])}</button>' for c in chats)}
  {"" if len(chats) < 2 else '<button class="chat-btn active" data-chat="all" onclick="selectChat(this)">All conversations</button>'}
</div>

<div id="tabBar" class="tabs"></div>
<div class="main" id="mainContent"></div>

<script>
const PRIMARY_USER = {_j(PRIMARY_USER_NAME)};
const CHATS = {_j(chat_data)};
const CROSS_CHAT = {_j(cross_chat)};
const PEOPLE = {_j(people_data)};
const PEOPLE_AI = {_j(people_ai)};
const TIMELINE_HIGHLIGHTS = {_j(ai_results.get("timeline_highlights", []))};
const CROSS_CHAT_LINKS = {_j(ai_results.get("cross_chat_links", []))};
const CRISIS_ASSESSED = {_j(ai_results.get("crisis_assessed", {}))};

let currentChat = null;
let currentTab = "conversations";
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
function makeLineChart(canvasId, labels, datasets) {{
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if(!ctx) return;
  charts[canvasId] = new Chart(ctx,{{
    type:"line",
    data:{{labels,datasets}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{position:"top"}},tooltip:{{mode:"index"}}}},
      scales:{{
        x:{{ticks:{{maxTicksLimit:12,maxRotation:45}}}},
        y:{{min:-1,max:1,ticks:{{stepSize:0.5}}}}
      }}
    }}
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
    {{id:"people",label:"People & Sentiment",showAll:true,showSingle:true}},
    {{id:"narrative",label:"Narrative vs Record",showAll:false,showSingle:true}},
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
function selectChat(btn) {{
  document.querySelectorAll(".chat-btn").forEach(b=>b.classList.remove("active"));
  btn.classList.add("active");
  currentChat=btn.dataset.chat;
  // Reset to appropriate tab
  if(currentChat==="all") currentTab="conversations"; else currentTab="timeline";
  renderTabBar(currentChat);
  renderMain();
}}

function selectTab(tabId) {{
  currentTab=tabId;
  document.querySelectorAll(".tab-btn").forEach(b=>b.classList.toggle("active",b.textContent.trim().replace(/⚠ /,"")===tabId||b.onclick.toString().includes(`'${{tabId}}'`)));
  // Re-render tab bar to update active class
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
      case "people": renderPeople(el); break;
      case "framework": renderFramework(el,chat); break;
      case "narrative": renderNarrative(el,chat); break;
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
    case "people": renderPeople(el); break;
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
      html+=`<div class="bubble ${{side}}">
        <div class="bubble-inner">${{esc(m.text)}}</div>
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

  // ── Mood chart ──────────────────────────────────────────────────────────
  const sentDatasets=[];
  const allDates=[...new Set(chats.flatMap(c=>Object.keys(c.daily_sentiment.user||{{}}).concat(Object.keys(c.daily_sentiment.contact||{{}}))))].sort();
  chats.forEach((chat,i)=>{{
    const userAvg=rollingAvg(chat.daily_sentiment.user||{{}});
    const contAvg=rollingAvg(chat.daily_sentiment.contact||{{}});
    const label=chats.length>1?chat.name:"";
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
  }});
  requestAnimationFrame(()=>makeLineChart("sentChart",allDates,sentDatasets));

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
      html+=`<div style="border:1px solid var(--danger);border-radius:var(--radius);padding:12px;margin-bottom:10px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${{fmt(f.date)}}</div>
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
  const names=Object.keys(PEOPLE);
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
    html+=`<div class="person-card">
      <div class="name">${{esc(name)}} <span style="font-size:12px;color:var(--muted);font-weight:400">×${{p.count}} mentions · ${{p.chats.join(", ")}}</span></div>
      ${{label?`<div class="label">${{esc(label)}}</div>`:""}}
      <div class="person-card sentiment-bar">
        <div class="sb-pos" style="width:${{sd.positive||0}}%"></div>
        <div class="sb-neu" style="width:${{sd.neutral||0}}%"></div>
        <div class="sb-neg" style="width:${{sd.negative||0}}%"></div>
      </div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:8px">+${{sd.positive||0}}% / =${{sd.neutral||0}}% / -${{sd.negative||0}}%</div>
      ${{warn?`<div style="background:#fef3c7;border-radius:4px;padding:8px;font-size:12px;margin-bottom:8px">⚠ ${{esc(warn)}}</div>`:""}}
      ${{summary?`<div class="summary">${{esc(summary)}}</div>`:""}}
      <textarea class="note" placeholder="Add your own notes about ${{name}}…" onchange="savePersonEdit('${{name}}',this.value)">${{esc(saved)}}</textarea>
      ${{p.excerpts&&p.excerpts.length?`<details style="margin-top:8px"><summary style="font-size:12px;color:var(--muted);cursor:pointer">Show message excerpts</summary><div style="margin-top:8px">
        ${{p.excerpts.map(e=>`<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px">"${{esc(e.slice(0,200))}}"</div>`).join("")}}
      </div></details>`:""}}</div>`;
  }});
  html+=`</div>`;
  el.innerHTML=html;
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
  // Auto-select: "all" if multiple chats, first chat otherwise
  const defaultBtn = CHATS.length>1
    ? document.querySelector('[data-chat="all"]')
    : document.querySelector('.chat-btn');
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

        # Call 6 — crisis assessment (Claude, optional)
        if use_claude_crisis:
            print("[AI] Crisis assessment (Claude API)…")
            for chat in chats:
                if chat["analytics"]["crisis_flags"]:
                    assessed = ai_crisis_assessment_claude(chat["analytics"]["crisis_flags"], api_key)
                    ai_results["crisis_assessed"][chat["contact_name"]] = {
                        i: item for i, item in enumerate(assessed)
                    }

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
