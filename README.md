# WhatsApp Chat Analyser

A local, private tool that processes WhatsApp chat exports and produces a single self-contained HTML analysis document. No server, no cloud upload, no external API required (Ollama runs locally).

## What it produces

- **Conversations overview** — initiation balance, visit summary, distress signal counts per contact
- **Message timeline** — chat bubbles grouped by day with intent badges and per-day notes (saved in your browser)
- **Visits & Availability** — offer → accept/cancel → happened chains, cancellation attribution
- **Sentiment chart** — rolling 7-day mood trend per sender, all contacts on one chart
- **Initiation balance** — who starts conversations, with trend direction
- **People & Sentiment** — cards for every person mentioned 3+ times, with AI-generated summaries and sentiment distribution
- **Narrative vs Record** — AI-generated contrast pairs between stated narrative and behavioural data
- **Communications Framework** — your framework file shown inline with AI-suggested data-backed adjustments
- **Crisis Moments** — click-to-reveal flagged messages; optional Claude API assessment; always shows Samaritans number

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

**NLP engine choice:**

| Engine | Quality | Install size | Setting |
|--------|---------|-------------|---------|
| `transformers` (default) | Higher — uses DistilBERT | ~300 MB download on first run | `NLP_ENGINE = "transformers"` |
| `textblob` | Lighter — rule-based | ~50 MB | `NLP_ENGINE = "textblob"` |

If you choose TextBlob, also run:
```bash
python -m textblob.download_corpora
```

### 2. Install Ollama (for AI features)

1. Download from **https://ollama.com**
2. Pull Gemma:
   ```bash
   ollama pull gemma3
   # or for the larger model:
   ollama pull gemma3:12b
   ```
3. Confirm it's running: `ollama list`

The analyser connects to Ollama at `http://localhost:11434` by default. All AI processing is local — nothing leaves your machine.

### 3. Export your WhatsApp chats

On iPhone or Android:
- Open a chat → tap the contact name → **Export Chat** → **Without Media**
- This produces a `.txt` file, e.g. `WhatsApp Chat with Mum.txt`

### 4. Configure `analyser.py`

Edit the `CONFIGURATION` section at the top of `analyser.py`:

```python
PRIMARY_USER_NAME = "Your Name Here"   # must match exactly how it appears in exports

CHATS = [
    {
        "file": "/path/to/WhatsApp Chat with Mum.txt",
        "contact_name": "Mum",
        "contact_relationship": "Mother",
        "tags": ["family"],
        "framework": "/path/to/mum_framework.md",  # or None
    },
    {
        "file": "/path/to/WhatsApp Chat with Work.txt",
        "contact_name": "James",
        "contact_relationship": "Colleague",
        "tags": ["work"],
        "framework": None,
    },
]
```

**Framework files** are optional. Copy `default_framework.md`, fill in the sections, and point `"framework"` at it. If omitted (or `None`), that tab is hidden for that chat.

### 5. Run

```bash
python analyser.py
```

Then open `analysis.html` in your browser.

---

## CLI flags

| Flag | Description |
|------|-------------|
| `--no-ai` | Skip all AI calls. Python-only output. |
| `--local-only` | Use Ollama only; skip Claude even for crisis assessment. |
| `--claude-crisis` | Enable Claude API for crisis assessment (requires `ANTHROPIC_API_KEY`). |
| `--chat NAME` | Process only one chat by contact name (e.g. `--chat "Mum"`). |
| `--output PATH` | Override the output file path (default: `analysis.html`). |
| `--log-tokens` | Print token usage per AI call at the end of the run. |

### Examples

```bash
# Python-only, no Ollama needed
python analyser.py --no-ai

# Single chat only
python analyser.py --chat "Mum" --output mum_analysis.html

# With Claude crisis assessment
python analyser.py --claude-crisis

# Check token usage
python analyser.py --log-tokens
```

---

## Framework files

Copy `default_framework.md` as a starting point:

```bash
cp default_framework.md mum_framework.md
```

Fill in the sections (they're plain text / Markdown — no special format required). The framework is displayed verbatim in the HTML, with AI-suggested adjustments shown inline below each section based on the actual conversation data.

---

## AI calls

All six calls are relationship-agnostic and refer to "the contact" internally. They run sequentially; if Ollama is unreachable, each call degrades gracefully to Python-only output with a note.

| Call | What it does | Trigger |
|------|-------------|---------|
| 1 | People & Sentiment card summaries | After Python people extraction |
| 2 | Narrative vs Record contrast pairs | After visit tracker + initiation analysis |
| 3 | Framework personalisation suggestions | Only if a framework file is configured |
| 4 | Timeline highlights | After full parse |
| 5 | Cross-chat link analysis | Only if more than one chat loaded |
| 6 | Crisis assessment | Optional; Claude API only; only if crisis flags found |

---

## Crisis Moments tab

- **Always private** — shown behind a click-to-reveal button
- **Not diagnostic** — these are keyword-based flags for personal review, not clinical assessment
- **AI assessment** (Call 6) is off by default. Enable with `--claude-crisis` or `CRISIS_AI_ASSESSMENT = True`
- If enabled, flagged excerpts are sent to Claude with no surrounding context and no names
- Always shows: *"If you are concerned about immediate safety, call 999 or Samaritans on 116 123 (free, 24/7)"*

---

## Privacy

- All processing is local
- Ollama runs on your machine — no data leaves it
- The output HTML is a local file — it contains your message data
- The Claude API call (crisis assessment only, opt-in) sends anonymised excerpts to Anthropic's API — no names, no context
- Browser notes and person card edits are stored in `localStorage` — local to your browser

---

## Troubleshooting

**"No messages parsed"**
- Check `PRIMARY_USER_NAME` exactly matches how your name appears in the export
- Try opening the `.txt` file in a text editor and checking the format of the first few lines

**"Ollama not reachable"**
- Run `ollama serve` in a terminal, or ensure the Ollama app is running
- Check `OLLAMA_BASE_URL = "http://localhost:11434"` in the config

**Sentiment model slow on first run**
- The DistilBERT model downloads (~250 MB) once and is cached. Subsequent runs are fast.
- Switch to `NLP_ENGINE = "textblob"` for a faster, lighter option.

**Framework tab not appearing**
- Ensure `"framework"` in the chat config points to a file that exists
- Check the path is absolute or relative to where you run the script

---

## Files

| File | Description |
|------|-------------|
| `analyser.py` | Main script |
| `requirements.txt` | Python dependencies |
| `default_framework.md` | Blank framework template — copy and fill in per relationship |
| `example_output.html` | Example report with anonymised synthetic data |
