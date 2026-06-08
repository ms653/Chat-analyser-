# Implementation Plan: SPEC.md → App

## Context

The app already has a working WhatsApp chat analyser (`analyser.py`, 2,147 lines) with:
a plaintext TXT parser, basic DistilBERT/TextBlob sentiment (positive/negative/neutral),
rule-based intent classification (visit states, distress, crisis keywords), Ollama/Gemma
AI calls, and a self-contained HTML report. SPEC.md was written to describe a substantially
richer analytics architecture. This plan maps every SPEC feature to the existing code and
schedules implementation in priority order.

---

## Gap Analysis: What SPEC Describes vs. What Exists

| SPEC Feature | Status | Lives In |
|---|---|---|
| Regex WhatsApp parser (multiline, LRM, delimiter) | ✅ Implemented (partially) | `analyser.py` `parse_chat()` |
| Ekman 6-emotion classification (DistilRoBERTa) | ❌ Missing | — |
| Emoji sentiment scoring (lexicon + hybrid formula) | ❌ Missing | — |
| DRIVE coping framework (positive/negative + sub-themes) | ❌ Missing | — |
| LDA + Dynamic Topic Modeling | ❌ Missing | `compute_topics()` is keyword-only |
| ABSA entity-sentiment mapping | ❌ Missing | `extract_person_mentions()` is heuristic only |
| Crisis Tier 1 (4-dimension keyword taxonomy) | ⚠️ Partial | `classify_intent()` — flat list only |
| Crisis Tier 2 (Random Forest / TF-IDF) | ❌ Missing | — |
| Crisis Tier 3 (Gemma context validation) | ⚠️ Partial | Claude API only (not Gemma) |
| NetworkX + Plotly topology visualisation | ❌ Missing | — |
| SQLite ingestion (msgstore.db / wa.db) | ❌ Missing | — |
| Audio transcription (Whisper + FFmpeg) | ❌ Missing | — |
| Programmatic regex class (ProgrammaticWhatsAppParser) | ❌ Not refactored | — |

**Out of scope for this plan** (high complexity, separate input source):
- SQLite database ingestion
- Audio transcription (Whisper + FFmpeg)
- Focal Loss model fine-tuning (training, not inference)

---

## Implementation Phases

### Phase 1 — Ekman Emotion Engine + Emoji Sentiment

**Goal:** Replace the current binary positive/negative sentiment with 7-category Ekman
emotion scoring plus emoji-aware composite valence.

**Files changed:** `analyser.py`, `requirements.txt`

**Key changes:**
- Replace `_load_sentiment_pipeline()` (line 204) and `score_sentiment_transformers()`
  (line 224) with a new `LocalAffectAggregator` class matching SPEC §2.1.
- Model: `j-hartmann/emotion-english-distilroberta-base` (download on first run, CPU-safe).
- Add emoji sentiment lexicon dict (😭→-0.85, 🥰→0.90, etc.) and the weighted hybrid:
  `composite_valence = 0.7 * (joy - sadness) + 0.3 * avg_emoji_score`
- `add_sentiment_scores()` output changes from `{"label": str, "score": float}` to:
  `{"emotions": {anger, disgust, fear, joy, neutral, sadness, surprise}, "composite_valence": float, "emoji_sentiment": float}`
- Update all downstream HTML rendering that reads `msg["sentiment"]["label"]` to use
  composite_valence and the emotions dict.
- Add `emoji>=2.2.0` to `requirements.txt`.
- Keep `textblob` fallback path; when `NLP_ENGINE="textblob"`, preserve existing behaviour.

**HTML output additions:**
- Sentiment chart replaces positive/negative with joy/sadness/anger/fear lines.
- Message bubbles show dominant emotion label badge (if score > 0.4).

---

### Phase 2 — DRIVE Coping Framework Classifier

**Goal:** Add a new analysis module that classifies each message into a positive or negative
coping sub-theme, and surface this as a new "Coping Dynamics" section in the report.

**Files changed:** `analyser.py`

**Key changes:**
- Add `classify_coping(msg_text)` function using a two-layer approach:
  1. Fast regex pass matching SPEC's keyword signals per sub-theme.
  2. For ambiguous messages (low regex confidence), queue for Gemma.
- Add a new Ollama AI call `ai_coping_analysis(chat)` — 7th AI call — that sends a
  sliding 5-message window to Gemma and receives structured JSON per SPEC §2.3:
  ```json
  {"dominant_strategy": "Positive|Negative|None",
   "sub_theme": "...",
   "explanation": "..."}
  ```
- Add `coping_summary` to each chat's computed metrics:
  `{"positive_pct": float, "negative_pct": float, "by_subtheme": Counter}`
- Reuse existing `_ollama_chat()` (line 706) for the Gemma call.

**HTML output additions:**
- New "Coping Dynamics" tab per chat with:
  - Donut chart: positive vs. negative coping ratio.
  - Sub-theme breakdown table.
  - Example messages per sub-theme.

---

### Phase 3 — Enhanced Crisis Detection (4-Dimension Taxonomy + Tier 2 ML)

**Goal:** Upgrade the existing flat crisis/distress keywords to the 4-dimension taxonomy
(Cognitive, Physiological, Behavioral, Emotional) with level assignments and add a
Random Forest Tier 2 classifier.

**Files changed:** `analyser.py`, `requirements.txt`

**Key changes:**

**Tier 1 upgrade:**
- Replace `CRISIS_FLAG` / `DISTRESS_SIGNAL` single-list patterns in `classify_intent()`
  (line 334) with a structured `CRISIS_TAXONOMY` dict keyed by dimension and level:
  ```python
  CRISIS_TAXONOMY = {
      "cognitive":     {"level": 2, "patterns": ["hopeless", "worthless", ...]},
      "physiological": {"level": 1, "patterns": ["can't sleep", "exhausted", ...]},
      "behavioral":    {"level": 3, "patterns": ["goodbye", "make a plan", ...]},
      "emotional":     {"level": 2, "patterns": ["crying", "rage", "panic", ...]},
  }
  ```
- Each flagged message gets `crisis_dimension` and `crisis_level` (1/2/3) fields.

**Tier 2 — Random Forest:**
- Add `train_crisis_classifier()` function. On first run, trains using a small bundled
  synthetic seed dataset (no external download needed — hand-labelled examples in code).
- Feature extraction: lowercase, strip URLs/usernames/numerics, normalize repeated chars,
  tokenise, remove stopwords (NLTK), stem (Porter), then `TfidfVectorizer`.
- Model: `sklearn.ensemble.RandomForestClassifier(n_estimators=100)`.
- Pickle the fitted model to `~/.whatsapp_analyser_crisis_model.pkl`; reload on subsequent
  runs.
- Add `scikit-learn>=1.3.0` and `nltk>=3.8` to `requirements.txt`.
- Tier 2 runs on every message that Tier 1 does not already flag at Level 3.

**Tier 3 upgrade — Gemma context validation:**
- Upgrade existing `ai_crisis_assessment_claude()` (line 877) to also offer a local path
  via Gemma using `LocalGemmaInferencePipeline` pattern from SPEC §4.3.
- When `--local-only` flag is set (or no Claude API key), route Level 2/3 flags through
  Gemma for sarcasm vs. literal disambiguation.
- Add `crisis_validator.is_literal_risk` bool to flagged message records.

---

### Phase 4 — Aspect-Based Sentiment Analysis (ABSA)

**Goal:** Upgrade `extract_person_mentions()` (currently capitalisation heuristic + simple
sentiment averaging) to true aspect-based entity–sentiment mapping.

**Files changed:** `analyser.py`, `requirements.txt`

**Key changes:**
- Add `run_absa(chat)` function with two tiers:
  1. **Gemma tier (default, no new deps):** Uses existing `_ollama_chat()` to send
     message batches to Gemma with the structured JSON schema from SPEC §3.2. Returns
     `[{target_entity, valence, reasoning}]` per message.
  2. **SetFit tier (optional, `--absa-model setfit` flag):** Uses `setfit` + `spacy`
     for CPU-optimised local inference.
- Replace the capitalisation heuristic in `extract_person_mentions()` with ABSA output.
- Update `people` records to carry `{valence_history: [float], avg_valence: float,
  aspect_excerpts: [str]}`.
- Add `spacy>=3.7.0` to requirements (needed for SetFit tier and NER).

**HTML output changes:**
- People cards now show continuous valence dial (−1.0 → +1.0) instead of
  positive/neutral/negative bar.
- Add reasoning excerpts toggle per entity.

---

### Phase 5 — LDA Topic Modelling (replacing keyword topics)

**Goal:** Replace the keyword-counting `compute_topics()` function with probabilistic
LDA discovery, and add time-windowed Dynamic Topic Modelling.

**Files changed:** `analyser.py`, `requirements.txt`

**Key changes:**
- Add `run_lda_topics(messages, n_topics=5)` using `gensim.models.LdaModel`.
- Preprocessing: tokenise, remove stopwords, lemmatise (spaCy, already added in Phase 4).
- Dynamic slice: split messages into monthly time windows; fit per-window LDA; track
  topic drift across windows.
- Keep legacy `CUSTOM_TOPICS` keyword matching as an override layer for user-defined terms.
- Add `gensim>=4.3.0` to requirements.

**HTML output additions:**
- New "Topics Over Time" section: stacked area chart per topic per month.
- Topic word cloud per discovered theme.

---

### Phase 6 — NetworkX + Plotly Topology Visualisation

**Goal:** Add the `InteractiveTopologyGenerator` from SPEC §5, producing an interactive
entity–sentiment network graph embedded in the report.

**Files changed:** `analyser.py`, `requirements.txt`

**Key changes:**
- Add `InteractiveTopologyGenerator` class (verbatim from SPEC §5.2 with minor
  integration adjustments):
  - Input: ABSA records from Phase 4 (`sender`, `entity`, `valence`).
  - `construct_interaction_graph()` → builds `nx.Graph` with edge weights + valence lists.
  - `generate_plotly_figure()` → spring layout, edge traces, node traces (coloured by
    avg valence, sized by degree centrality), Louvain community detection.
- Serialise the Plotly figure to JSON and embed in the HTML report.
- Add `networkx>=3.2`, `plotly>=5.20.0`, `python-louvain>=0.16` to requirements.

**HTML output additions:**
- New "Relationship Network" tab containing the interactive Plotly graph.
- Hover tooltips show entity name, centrality, avg valence.

---

### Phase 7 — Parser Refactor (ProgrammaticWhatsAppParser)

**Goal:** Refactor the existing regex parser to the class-based `ProgrammaticWhatsAppParser`
approach from SPEC §6.1, improving format coverage and testability.

**Files changed:** `analyser.py`

**Key changes:**
- Wrap existing `MSG_RE`, `_parse_dt()`, `parse_chat()` into the new class.
- Add `generate_regex_patterns()` accepting a simplified `hformat` string
  (e.g., `"%d/%m/%y, %H:%M - %name:"`) and translating it to a compilable regex via the
  `regex_map` lookup table.
- Retain existing auto-detection fallback for users who don't specify a format.
- Update `gui.py` to expose an optional "Date format" field in the config form.
- This phase is lower risk since the core regex logic is preserved; it's a structural wrap.

---

### Phase 8 — Deterministic Test Suite

**Goal:** Add a pytest suite covering the critical parsing and analysis paths (per CLAUDE.md
guidelines).

**Files added:** `tests/test_parser.py`, `tests/test_affect.py`, `tests/test_crisis.py`

**Key test cases:**
- iOS LRM `‎` stripping in parser.
- Multiline message concatenation.
- AM/PM 12-hour format parsing.
- Emoji composite valence calculation.
- Crisis taxonomy dimension/level assignment.
- Coping framework regex classification.

---

## Critical Files

| File | Role in Implementation |
|---|---|
| `analyser.py` | All analysis logic; every phase touches this file |
| `gui.py` | Phase 7 only: adds "Date format" config field |
| `requirements.txt` | Updated each phase for new deps |

---

## New Dependencies by Phase

| Phase | New Packages |
|---|---|
| 1 | `emoji>=2.2.0` |
| 3 | `scikit-learn>=1.3.0`, `nltk>=3.8` |
| 4 | `spacy>=3.7.0` (+ `en_core_web_sm` model) |
| 5 | `gensim>=4.3.0` |
| 6 | `networkx>=3.2`, `plotly>=5.20.0`, `python-louvain>=0.16` |

---

## Verification

After each phase, verify as follows:

1. **Phase 1**: Run `python analyser.py --no-ai` on a sample chat; confirm the output HTML
   shows 7 emotion scores per message bubble and a multi-line sentiment chart.
2. **Phase 2**: Run with Ollama active; open "Coping Dynamics" tab and confirm donut chart
   renders with sub-theme breakdown.
3. **Phase 3**: Inject synthetic crisis messages into a test chat; confirm Level 1/2/3
   assignments appear in "Crisis Moments" tab with dimension labels.
4. **Phase 4**: Check "People & Sentiment" cards show valence dial and reasoning excerpts
   from Gemma.
5. **Phase 5**: Open "Topics Over Time" and verify monthly stacked chart with auto-labelled
   topics (not hard-coded keywords).
6. **Phase 6**: Open "Relationship Network" tab; confirm interactive Plotly graph loads with
   hover tooltips.
7. **Phase 7**: Feed an iOS export with `[DD/MM/YYYY, HH:MM:SS]` format; confirm all
   messages parse correctly with no timestamp misses.
8. **Phase 8**: Run `pytest` — all tests green.
