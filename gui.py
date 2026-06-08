#!/usr/bin/env python3
"""
WhatsApp Chat Analyser — GUI
Double-click the built .app to launch. No terminal or browser needed.
The config form and analysis output both live inside this one window.
"""

import os
import re
import sys
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import webview

# ── Path fix for PyInstaller bundle vs running from source ──────────────────
if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS  # PyInstaller extracts here
    # Windowed .app has no terminal — redirect stdout/stderr to a log file
    # so print() calls don't raise BrokenPipeError and crash background threads
    _log = open(Path.home() / ".whatsapp_analyser.log", "a", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

import analyser  # noqa: E402  (must come after path fix)

# Saved config lives in home dir so it persists between runs
CONFIG_FILE   = Path.home() / ".whatsapp_analyser_config.json"
# Notes saved separately so they survive HTML regeneration
NOTES_FILE    = Path.home() / ".whatsapp_analyser_notes.json"
# Project source directory — saved on first source run, read back when frozen
_PROJ_PATH_FILE = Path.home() / ".whatsapp_analyser_project.txt"

GITHUB_REPO = "ms653/Chat-analyser-"


def _get_build_sha() -> str:
    """Return the git SHA this .app was compiled from, or '' when running from source."""
    if not getattr(sys, "frozen", False):
        return ""
    try:
        return (Path(_BASE) / "build_sha.txt").read_text().strip()
    except Exception:
        return ""


def _get_project_dir() -> Path | None:
    """
    Return the source project directory regardless of whether we are running
    from source or as a frozen .app bundle.
    """
    if not getattr(sys, "frozen", False):
        # Running from source — the directory containing this file
        p = Path(os.path.abspath(__file__)).parent
        try:
            _PROJ_PATH_FILE.write_text(str(p))
        except Exception:
            pass
        return p

    # Frozen .app — try saved path first
    try:
        if _PROJ_PATH_FILE.exists():
            p = Path(_PROJ_PATH_FILE.read_text().strip())
            if p.exists() and (p / "gui.py").exists():
                return p
    except Exception:
        pass

    # Fall back: derive from the .app bundle path
    # sys.executable → .../dist/WhatsApp Analyser.app/Contents/MacOS/binary
    try:
        p = Path(sys.executable).resolve().parents[4]
        if (p / "gui.py").exists():
            return p
    except Exception:
        pass

    return None

# ── Back-button toolbar injected into results HTML ───────────────────────────
_TOOLBAR = (
    '<div id="_gui_bar" style="position:fixed;top:0;left:0;right:0;height:44px;'
    'background:#1a202c;display:flex;align-items:center;padding:0 20px;z-index:10000;'
    'box-shadow:0 2px 10px rgba(0,0,0,.4)">'
    '<button onclick="pywebview.api.go_back()" style="background:transparent;border:1px solid #4a5568;'
    'color:#e2e8f0;padding:5px 14px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500">'
    "← New Analysis</button>"
    '<span style="color:#4a5568;font-size:13px;margin-left:16px;font-family:-apple-system,sans-serif">'
    "WhatsApp Chat Analyser</span>"
    "</div>"
    '<div style="height:44px"></div>'
)


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON API (exposed to the JS config form)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyserAPI:
    """All methods on this class are callable from JavaScript as pywebview.api.*"""

    def __init__(self):
        self._window = None
        self._chats = None          # list of parsed chat dicts, set after analysis
        self._ollama_base = ""
        self._ollama_model = ""

    def set_window(self, w):
        self._window = w

    # ── File pickers ─────────────────────────────────────────────────────────

    def pick_chat_file(self):
        r = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Text Files (*.txt)", "All Files (*.*)"),
        )
        return r[0] if r else ""

    def pick_framework_file(self):
        r = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Markdown (*.md)", "Text Files (*.txt)", "All Files (*.*)"),
        )
        return r[0] if r else ""

    # ── Config persistence ───────────────────────────────────────────────────

    def load_config(self):
        try:
            if CONFIG_FILE.exists():
                return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
        return None

    def save_config(self, config):
        try:
            CONFIG_FILE.write_text(json.dumps(config, indent=2))
        except Exception as e:
            print(f"[WARN] Config save failed: {e}")

    # ── Chat file helpers ────────────────────────────────────────────────────

    def detect_senders(self, file_path: str) -> list:
        """Return sender names found in the chat file, most frequent first."""
        return analyser.detect_senders(file_path)

    def get_ollama_models(self, url: str) -> list:
        """Query Ollama at the given URL and return a list of installed model names."""
        try:
            import requests as _req
            r = _req.get(f"{url.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as e:
            return []

    # ── Notes persistence (disk) ─────────────────────────────────────────────
    # Called from JS in the results HTML so notes survive browser storage resets
    # and HTML regenerations. Keyed by noteKey(chatId, date) strings.

    def save_note(self, key: str, value: str):
        try:
            notes = {}
            if NOTES_FILE.exists():
                notes = json.loads(NOTES_FILE.read_text())
            notes[key] = value
            NOTES_FILE.write_text(json.dumps(notes, indent=2))
        except Exception as e:
            print(f"[WARN] Note save failed: {e}")

    def load_notes(self):
        try:
            if NOTES_FILE.exists():
                return json.loads(NOTES_FILE.read_text())
        except Exception:
            pass
        return {}

    # ── Navigation ───────────────────────────────────────────────────────────

    def go_back(self):
        """Return to the config form from the results view."""
        self._window.load_html(CONFIG_HTML)

    # ── Live AI — Chat Q&A ───────────────────────────────────────────────────

    def chat_qa(self, chat_id: str, question: str) -> str:
        """Answer a question about a specific chat (or all chats) using Ollama."""
        if not self._chats:
            return "No chat data available. Please run an analysis first."
        if not self._ollama_base or not self._ollama_model:
            return "Ollama not configured. Please run an analysis first."

        q_lower = question.lower()
        q_words = [w for w in q_lower.split() if len(w) > 2]

        # Gather candidate messages
        if chat_id == "all":
            all_msgs = [
                {**m, "_chat": c["contact_name"]}
                for c in self._chats
                for m in c["messages"]
            ]
        else:
            chat = next(
                (c for c in self._chats
                 if re.sub(r"\W+", "_", c["contact_name"]) == chat_id
                 or c["contact_name"] == chat_id),
                None,
            )
            if not chat:
                return f"Chat '{chat_id}' not found."
            all_msgs = [{**m, "_chat": chat["contact_name"]} for m in chat["messages"]]

        # Rank by keyword match count, break ties by recency (index)
        def _score(item):
            idx, m = item
            text_lower = (m.get("text") or "").lower()
            match_count = sum(1 for w in q_words if w in text_lower)
            return (match_count, idx)  # higher idx = more recent

        ranked = sorted(enumerate(all_msgs), key=_score, reverse=True)
        top_msgs = [m for _, m in ranked[:40]]
        # Sort selected messages chronologically for readability
        top_msgs.sort(key=lambda m: str(m.get("date") or ""))

        # Build prompt
        chat_label = "all conversations" if chat_id == "all" else chat_id
        lines = []
        for m in top_msgs:
            chat_prefix = f"[{m['_chat']}] " if chat_id == "all" else ""
            lines.append(f"[{m.get('date')} {m.get('sender')}] {chat_prefix}{(m.get('text') or '')[:300]}")

        prompt = (
            f"You are answering a question about the following WhatsApp messages from {chat_label}.\n\n"
            f"Question: {question}\n\n"
            f"Relevant messages:\n" + "\n".join(lines) + "\n\n"
            "Answer concisely and specifically, referencing what the messages actually say. "
            "If you cannot answer from the messages provided, say so."
        )
        result = analyser._ollama_chat(
            [{"role": "user", "content": prompt}],
            self._ollama_base,
            self._ollama_model,
            json_mode=False,
        )
        return result or "No response from Ollama."

    # ── Live AI — Mood Explainer ─────────────────────────────────────────────

    def explain_period(self, chat_id: str, date_str: str) -> str:
        """Explain the emotional tone of messages around a given date."""
        import datetime as _dt
        if not self._chats:
            return "No chat data available. Please run an analysis first."
        if not self._ollama_base or not self._ollama_model:
            return "Ollama not configured. Please run an analysis first."

        try:
            centre = _dt.date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return f"Invalid date format: {date_str}"

        window_days = 14

        # Collect messages from the ±14-day window
        window_msgs = []
        for c in self._chats:
            if chat_id not in ("all",) and \
               re.sub(r"\W+", "_", c["contact_name"]) != chat_id and \
               c["contact_name"] != chat_id:
                continue
            for m in c["messages"]:
                try:
                    msg_date = _dt.date.fromisoformat(str(m.get("date") or ""))
                    if abs((msg_date - centre).days) <= window_days:
                        window_msgs.append({**m, "_chat": c["contact_name"]})
                except (ValueError, TypeError):
                    pass

        if not window_msgs:
            return f"No messages found within {window_days} days of {date_str}."

        # Sort chronologically, cap at 60
        window_msgs.sort(key=lambda m: str(m.get("date") or ""))
        window_msgs = window_msgs[:60]

        lines = [
            f"[{m.get('date')} {m.get('sender')}]: {(m.get('text') or '')[:250]}"
            for m in window_msgs
        ]
        prompt = (
            f"Here are WhatsApp messages from around {date_str} "
            f"(±{window_days} days). What themes or events seem to be driving the emotional "
            "tone during this period? Be specific — reference what is actually being discussed "
            "in the messages, not generic observations.\n\n"
            "Messages:\n" + "\n".join(lines)
        )
        result = analyser._ollama_chat(
            [{"role": "user", "content": prompt}],
            self._ollama_base,
            self._ollama_model,
            json_mode=False,
        )
        return result or "No response from Ollama."

    # ── Updates ──────────────────────────────────────────────────────────────

    def get_version(self) -> dict:
        build_sha = _get_build_sha()
        proj = _get_project_dir()
        if not proj or not shutil.which("git"):
            return {"hash": build_sha[:7] if build_sha else "unknown", "date": "unknown"}
        try:
            ref = build_sha if build_sha else "HEAD"
            r = subprocess.run(
                ["git", "log", "-1", "--format=%h|%cd", "--date=short", ref],
                capture_output=True, text=True, cwd=str(proj), timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split("|")
                return {"hash": parts[0], "date": parts[1] if len(parts) > 1 else ""}
        except Exception:
            pass
        return {"hash": build_sha[:7] if build_sha else "unknown", "date": "unknown"}

    def check_for_updates(self) -> dict:
        proj = _get_project_dir()
        if not proj:
            return {"error": "Project folder not found. Run python3 gui.py once first."}
        if not shutil.which("git"):
            return {"error": "git not found. Install Xcode Command Line Tools via the App Store."}
        try:
            subprocess.run(
                ["git", "fetch", "origin", "main"],
                capture_output=True, cwd=str(proj), timeout=15,
            )
            local = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=str(proj),
            ).stdout.strip()
            remote = subprocess.run(
                ["git", "rev-parse", "origin/main"],
                capture_output=True, text=True, cwd=str(proj),
            ).stdout.strip()

            # Pull any remote commits that haven't landed locally yet
            if local != remote:
                subprocess.run(
                    ["git", "pull", "origin", "main"],
                    capture_output=True, cwd=str(proj), timeout=30,
                )
                local = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, cwd=str(proj),
                ).stdout.strip()

            # Compare the running .app's baked-in SHA against the current source
            build_sha = _get_build_sha()
            if build_sha and build_sha != local:
                log = subprocess.run(
                    ["git", "log", "--oneline", f"{build_sha}..HEAD"],
                    capture_output=True, text=True, cwd=str(proj),
                ).stdout.strip()
                count = len(log.splitlines()) if log else 1
                return {"available": True, "count": count, "preview": log[:300]}

            # Running from source, or app already built from latest
            return {"available": False, "message": "You're already on the latest version."}
        except subprocess.TimeoutExpired:
            return {"error": "Timed out — check your internet connection."}
        except Exception as e:
            return {"error": str(e)}

    def do_update(self):
        """Pull latest code and rebuild the .app. Runs in background thread."""
        t = threading.Thread(target=self._update_worker, daemon=True)
        t.start()

    def _ulog(self, msg: str):
        self._window.evaluate_js(f"updateLog({json.dumps(str(msg))})")

    def _update_worker(self):
        proj = _get_project_dir()
        if not proj:
            self._ulog("ERROR: Project folder not found.")
            return
        try:
            # 1 — Pull (already done in check_for_updates, but safe to repeat)
            self._ulog("Pulling latest code from GitHub…")
            r = subprocess.run(
                ["git", "pull", "origin", "main"],
                capture_output=True, text=True, cwd=str(proj), timeout=30,
            )
            if r.returncode != 0:
                self._ulog(f"Git pull failed: {r.stderr.strip()}")
                return
            self._ulog(r.stdout.strip() or "Already at latest — rebuilding.")

            # 2 — Dependencies
            self._ulog("Checking dependencies…")
            req = proj / "requirements.txt"
            if req.exists():
                r2 = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
                    capture_output=True, text=True, timeout=120,
                )
                self._ulog("Dependencies up to date." if r2.returncode == 0
                           else f"Dependency warning: {r2.stderr[:200]}")

            # 3 — Write build SHA so the new .app knows what it was built from
            sha_file = proj / "build_sha.txt"
            try:
                current_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, cwd=str(proj),
                ).stdout.strip()
                sha_file.write_text(current_sha)
            except Exception:
                pass

            # 4 — Rebuild .app
            self._ulog("Rebuilding app — this takes about a minute…")
            r3 = subprocess.run(
                [
                    sys.executable, "-m", "PyInstaller",
                    "--windowed", "--onedir",
                    "--name", "WhatsApp Analyser",
                    "--hidden-import", "webview",
                    "--hidden-import", "webview.platforms.cocoa",
                    "--collect-all", "webview",
                    "--add-data", "build_sha.txt:.",
                    "--noconfirm",
                    str(proj / "gui.py"),
                ],
                capture_output=True, text=True, cwd=str(proj), timeout=300,
            )
            try:
                sha_file.unlink()
            except Exception:
                pass

            if r3.returncode != 0:
                self._ulog(f"Build failed:\n{r3.stderr[-600:]}")
                return

            new_app = proj / "dist" / "WhatsApp Analyser.app"
            if new_app.exists():
                self._ulog("✓ Update complete!")
                self._window.evaluate_js("updateDone()")
            else:
                self._ulog("Build finished but app not found — check dist/ folder.")
        except subprocess.TimeoutExpired:
            self._ulog("Timed out during update.")
        except Exception as e:
            self._ulog(f"Update error: {e}")

    def relaunch(self):
        """Open the freshly built .app and close this window."""
        proj = _get_project_dir()
        if proj:
            new_app = proj / "dist" / "WhatsApp Analyser.app"
            if new_app.exists():
                subprocess.Popen(["open", str(new_app)])
        self._window.destroy()

    # ── Analysis ─────────────────────────────────────────────────────────────

    def run_analysis(self, config):
        """Kick off analysis in a background thread (returns immediately to JS)."""
        t = threading.Thread(target=self._worker, args=(config,), daemon=True)
        t.start()

    # ── Background worker ─────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._window.evaluate_js(f"appendLog({json.dumps(str(msg))})")

    def _worker(self, config: dict):
        try:
            # ── Patch analyser module globals from GUI config ────────────────
            analyser.PRIMARY_USER_NAME = config.get("primary_user", "")
            analyser.OLLAMA_BASE_URL   = config.get("ollama_url", "http://localhost:11434")
            analyser.OLLAMA_MODEL      = config.get("ollama_model", "gemma4:e4b")
            analyser.ANTHROPIC_API_KEY = config.get("api_key", "")

            no_ai         = config.get("no_ai", False)
            claude_crisis = config.get("claude_crisis", False)
            engine        = config.get("nlp_engine", "textblob")
            custom_topics = config.get("custom_topics", [])

            # Store Ollama config for live API calls
            self._ollama_base  = config.get("ollama_url", "http://localhost:11434")
            self._ollama_model = config.get("ollama_model", "gemma4:e4b")

            # 1 ── Parse chats ────────────────────────────────────────────────
            chats_cfg = config.get("chats", [])
            self._log(f"Parsing {len(chats_cfg)} chat file(s)…")

            chats = []
            for cfg in chats_cfg:
                self._log(f"  → {cfg.get('contact_name', '?')}")
                c = analyser.parse_chat(cfg)
                if c["messages"]:
                    chats.append(c)
                else:
                    self._log(
                        f"    ⚠ No messages found — check the file path and that your "
                        f"name matches the export exactly"
                    )

            if not chats:
                self._window.evaluate_js(
                    "setError('No messages parsed. Check your chat file paths and "
                    "that your Primary User Name matches the export exactly.')"
                )
                self._window.evaluate_js("setRunning(false)")
                return

            total = sum(len(c["messages"]) for c in chats)
            self._log(f"Parsed {total:,} messages across {len(chats)} chat(s).")

            # 2 ── Per-chat NLP analysis ──────────────────────────────────────
            for chat in chats:
                self._log(f"Scoring sentiment — {chat['contact_name']}…")
                analyser.run_per_chat_analysis(chat, engine, custom_topics)

            # 3 ── Cross-chat correlation ─────────────────────────────────────
            self._log("Cross-chat correlation…")
            cross_chat = analyser.run_cross_chat_analysis(chats)

            # 4 ── AI calls ───────────────────────────────────────────────────
            ai_results = {
                "people_cards":        {},
                "narrative_vs_record": {},
                "framework_suggestions": {},
                "timeline_highlights": [],
                "cross_chat_links":    [],
                "crisis_assessed":     {},
                "relationship_summary": {},
            }

            if not no_ai:
                base  = analyser.OLLAMA_BASE_URL
                model = analyser.OLLAMA_MODEL

                # Quick sanity-check: ping Ollama and confirm the model exists
                _ollama_ok = False
                try:
                    import requests as _req
                    _r = _req.get(f"{base}/api/tags", timeout=5)
                    _models = [m["name"] for m in _r.json().get("models", [])]
                    _base_names = [m.split(":")[0] for m in _models]
                    if model in _models or model.split(":")[0] in _base_names:
                        _ollama_ok = True
                    else:
                        self._log(
                            f"⚠ Ollama is running but model '{model}' not found. "
                            f"Available: {', '.join(_models)}. "
                            f"Update the model name in Settings and re-run."
                        )
                except Exception as _e:
                    self._log(f"⚠ Cannot reach Ollama at {base} — AI features skipped. ({_e})")

                if not _ollama_ok:
                    no_ai = True

                self._log("AI ① People & sentiment cards…")
                ai_results["people_cards"] = analyser.ai_people_cards(
                    cross_chat["merged_people"], base, model
                )

                for chat in chats:
                    self._log(f"AI ② Narrative vs record — {chat['contact_name']}…")
                    ai_results["narrative_vs_record"][chat["contact_name"]] = (
                        analyser.ai_narrative_vs_record(chat, cross_chat, base, model)
                    )

                for chat in chats:
                    if chat.get("framework_content"):
                        self._log(f"AI ③ Framework suggestions — {chat['contact_name']}…")
                        ai_results["framework_suggestions"][chat["contact_name"]] = (
                            analyser.ai_framework_personalisation(chat, base, model)
                        )

                self._log("AI ④ Timeline highlights…")
                ai_results["timeline_highlights"] = analyser.ai_timeline_highlights(
                    chats, base, model
                )

                if len(chats) > 1:
                    self._log("AI ⑤ Cross-chat links…")
                    ai_results["cross_chat_links"] = analyser.ai_cross_chat_links(
                        cross_chat, chats, base, model
                    )

                if claude_crisis and analyser.ANTHROPIC_API_KEY:
                    self._log("AI ⑥ Crisis assessment (Claude)…")
                    for chat in chats:
                        flags = chat["analytics"]["crisis_flags"]
                        if flags:
                            assessed = analyser.ai_crisis_assessment_claude(
                                flags, analyser.ANTHROPIC_API_KEY
                            )
                            ai_results["crisis_assessed"][chat["contact_name"]] = {
                                i: item for i, item in enumerate(assessed)
                            }

                for chat in chats:
                    self._log(f"AI ⑨ Relationship summary — {chat['contact_name']}…")
                    ai_results["relationship_summary"][chat["contact_name"]] = (
                        analyser.ai_relationship_summary(chat, base, model)
                    )

            # Store chats for live Q&A / mood explainer calls
            self._chats = chats

            # 5 ── Generate HTML → temp file → read back ──────────────────────
            self._log("Building report…")
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            )
            tmp_path = tmp.name
            tmp.close()

            analyser.generate_html(chats, cross_chat, ai_results, tmp_path)
            html = Path(tmp_path).read_text(encoding="utf-8")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            # Inject the "← New Analysis" toolbar
            html = html.replace("<body>", "<body>" + _TOOLBAR, 1)

            self._log("Done ✓  Loading results…")
            self._window.evaluate_js("setRunning(false)")
            self._window.load_html(html)

        except Exception as exc:
            import traceback
            self._window.evaluate_js(f"setError({json.dumps(str(exc))})")
            self._window.evaluate_js("setRunning(false)")
            print(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG HTML  (the setup form — embedded so the .app is fully self-contained)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhatsApp Chat Analyser</title>
<style>
:root{
  --bg:#f0f4f8;--surface:#fff;--border:#e2e8f0;--text:#1a202c;--muted:#718096;
  --green:#25D366;--blue:#1a73e8;--danger:#ef4444;--radius:10px;
  --shadow:0 2px 8px rgba(0,0,0,.08);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;color:var(--text);line-height:1.5}
.app{max-width:740px;margin:0 auto;padding:24px 16px 80px}
.header{text-align:center;padding:36px 0 28px}
.header .icon{font-size:40px;line-height:1;margin-bottom:10px}
.header h1{font-size:24px;font-weight:800;margin-bottom:4px}
.header p{color:var(--muted);font-size:13px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:22px;margin-bottom:14px;box-shadow:var(--shadow)}
.card-head{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.step{width:26px;height:26px;border-radius:50%;background:var(--green);color:#fff;font-weight:700;font-size:13px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.card-head h2{font-size:15px;font-weight:700}
label{display:block;font-size:13px;font-weight:600;color:var(--text);margin-bottom:5px}
.hint{font-size:12px;color:var(--muted);margin-top:3px}
input[type=text],input[type=password]{width:100%;padding:9px 12px;border:1.5px solid var(--border);border-radius:7px;font-size:14px;color:var(--text);background:var(--bg);outline:none;transition:border .15s}
input[type=text]:focus,input[type=password]:focus{border-color:var(--blue);background:#fff}
.file-row{display:flex;gap:8px}
.file-row input{flex:1;cursor:default;font-size:13px}
.btn{display:inline-flex;align-items:center;gap:5px;padding:8px 16px;border-radius:7px;border:none;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s}
.btn-ghost{background:var(--bg);color:var(--text);border:1.5px solid var(--border)}
.btn-ghost:hover{background:#e9ecef}
.btn-dashed{background:transparent;color:var(--blue);border:1.5px dashed #93c5fd;width:100%;padding:10px;border-radius:7px;font-size:13px;margin-top:10px}
.btn-dashed:hover{background:#eff6ff}
.btn-remove{background:transparent;color:#ef4444;border:1px solid #fecaca;padding:4px 10px;font-size:12px;border-radius:5px}
.btn-remove:hover{background:#fee2e2}
.chat-row{border:1.5px solid var(--border);border-radius:8px;padding:16px;margin-bottom:12px;background:var(--bg)}
.chat-row-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.chat-row-head .lbl{font-weight:700;font-size:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.full{grid-column:1/-1}
.fld{display:flex;flex-direction:column;gap:4px}
.radio-group{display:flex;gap:20px;flex-wrap:wrap;padding:4px 0}
.radio-opt{display:flex;align-items:center;gap:7px;font-size:14px;cursor:pointer;font-weight:400}
.radio-opt input{accent-color:var(--blue)}
.check-opt{display:flex;align-items:center;gap:8px;font-size:14px;cursor:pointer;font-weight:400;padding:6px 0}
.check-opt input{accent-color:var(--blue);width:15px;height:15px}
details summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px;padding:4px 0;user-select:none}
details summary::before{content:"▶";font-size:9px;color:var(--muted);transition:transform .2s}
details[open] summary::before{transform:rotate(90deg)}
.detail-body{padding-top:14px}
.run-wrap{text-align:center;padding:8px 0}
.run-btn{padding:14px 48px;font-size:16px;font-weight:800;border-radius:12px;background:var(--green);color:#fff;border:none;cursor:pointer;box-shadow:0 4px 14px rgba(37,211,102,.35);transition:all .15s}
.run-btn:hover:not(:disabled){background:#1db854;box-shadow:0 6px 18px rgba(37,211,102,.45);transform:translateY(-1px)}
.run-btn:disabled{background:#a0aec0;cursor:not-allowed;box-shadow:none;transform:none}
.err{background:#fef2f2;border:1.5px solid #fecaca;border-radius:7px;padding:12px 16px;margin-top:12px;font-size:13px;color:var(--danger);display:none}
.log-wrap{background:#0f172a;border-radius:var(--radius);padding:16px;margin-top:14px;display:none}
.log-title{color:#4ade80;font-family:monospace;font-size:12px;margin-bottom:8px;font-weight:700}
#log{max-height:220px;overflow-y:auto;font-family:"SF Mono",monospace;font-size:12px;color:#94a3b8;line-height:1.9}
.log-ok{color:#4ade80}
.log-warn{color:#fbbf24}
#apiField{display:none;margin-top:10px}
@media(max-width:540px){.grid2{grid-template-columns:1fr}.full{grid-column:1}}
</style>
</head>
<body>
<div class="app">

  <div class="header">
    <div class="icon">💬</div>
    <h1>WhatsApp Chat Analyser</h1>
    <p>Fully local &amp; private — nothing leaves your machine</p>
  </div>

  <!-- Step 1 — Your name -->
  <div class="card">
    <div class="card-head"><div class="step">1</div><h2>Your name</h2></div>
    <div class="fld">
      <label for="primaryUser">Name exactly as it appears in your chat exports</label>
      <input type="text" id="primaryUser" placeholder="e.g. Morgan Strutton" autocomplete="off" spellcheck="false">
      <p class="hint">Open a .txt export in a text editor — your name appears at the start of your own messages.</p>
    </div>
  </div>

  <!-- Step 2 — Chat files -->
  <div class="card">
    <div class="card-head"><div class="step">2</div><h2>Chat files</h2></div>
    <div id="chatList"></div>
    <button class="btn-dashed" onclick="addChat()">+ Add chat</button>
  </div>

  <!-- Step 3 — Settings -->
  <div class="card">
    <div class="card-head"><div class="step">3</div><h2>Settings</h2></div>

    <details style="margin-bottom:14px" id="ollamaDetails">
      <summary>Ollama (local AI)</summary>
      <div class="detail-body">
        <div class="fld" style="margin-bottom:12px">
          <label for="ollamaUrl">Ollama URL</label>
          <div style="display:flex;gap:8px;align-items:center">
            <input type="text" id="ollamaUrl" value="http://localhost:11434" style="flex:1" oninput="scheduleModelRefresh()">
            <button class="btn btn-ghost" style="white-space:nowrap;font-size:13px" onclick="refreshOllamaModels()" id="ollamaRefreshBtn">⟳ Detect models</button>
          </div>
        </div>
        <div class="fld">
          <label for="ollamaModel">Model</label>
          <select id="ollamaModel" style="width:100%;padding:8px 10px;border:1.5px solid var(--border);border-radius:var(--radius);font-size:14px;background:var(--surface);color:var(--text)">
            <option value="">— click Detect models —</option>
          </select>
          <p class="hint" id="ollamaHint" style="margin-top:4px"></p>
        </div>
      </div>
    </details>

    <details>
      <summary>Advanced AI options</summary>
      <div class="detail-body">
        <label class="check-opt"><input type="checkbox" id="noAi"> Skip all AI — Python analysis only (no Ollama needed)</label>
        <label class="check-opt"><input type="checkbox" id="claudeCrisis" onchange="toggleKey()"> Use Claude API for crisis assessment <span style="color:var(--muted);font-size:12px">(opt-in, requires API key)</span></label>
        <div id="apiField">
          <label for="apiKey">Anthropic API key</label>
          <input type="password" id="apiKey" placeholder="sk-ant-…" autocomplete="off">
        </div>
      </div>
    </details>

    <details id="updateSection">
      <summary>Updates</summary>
      <div class="detail-body">
        <div style="font-size:13px;color:var(--muted);margin-bottom:10px">
          Version: <span id="versionLabel" style="font-family:monospace">…</span>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
          <button class="btn btn-ghost" style="font-size:13px" onclick="checkForUpdates()">🔍 Check for updates</button>
          <button class="btn btn-ghost" style="font-size:13px;display:none" id="doUpdateBtn" onclick="doUpdate()">⬇ Download &amp; install</button>
          <button class="btn btn-ghost" style="font-size:13px;display:none" id="relaunchBtn" onclick="relaunch()">🔄 Relaunch new version</button>
        </div>
        <div id="updateStatus" style="font-size:13px;color:var(--muted);margin-bottom:8px"></div>
        <div id="updateLogWrap" style="display:none;background:#0f172a;border-radius:8px;padding:12px">
          <div id="updateLog" style="font-family:monospace;font-size:12px;color:#94a3b8;max-height:180px;overflow-y:auto;line-height:1.8"></div>
        </div>
      </div>
    </details>
  </div>

  <!-- Run -->
  <div class="run-wrap">
    <button class="run-btn" id="runBtn" onclick="run()">▶ Run Analysis</button>
  </div>

  <div class="err" id="errBox"></div>

  <div class="log-wrap" id="logWrap">
    <div class="log-title">● Running…</div>
    <div id="log"></div>
  </div>

</div>
<script>
// ── Utilities ────────────────────────────────────────────────────────────────
let _chatId = 0;

function ea(s){ return (s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;'); }

// ── Chat rows ─────────────────────────────────────────────────────────────────
function chatHTML(id, d) {
  d = d || {};
  return `<div class="chat-row" id="cr-${id}">
    <div class="chat-row-head">
      <span class="lbl">Chat ${id+1}</span>
      <button class="btn btn-remove" onclick="rmChat(${id})">✕ Remove</button>
    </div>
    <div class="grid2">
      <div class="fld full">
        <label>WhatsApp export file (.txt)</label>
        <div class="file-row">
          <input type="text" id="f-${id}" placeholder="/path/to/WhatsApp Chat.txt" readonly value="${ea(d.file||'')}">
          <button class="btn btn-ghost" onclick="pick(${id},'chat')">Browse…</button>
        </div>
        <div id="sh-${id}" style="display:none;margin-top:10px;padding:12px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;font-size:13px">
          <div id="sh-lbl-${id}" style="font-weight:600;margin-bottom:6px">Senders detected — which one is you?</div>
          <div id="sb-${id}" style="display:flex;gap:8px;flex-wrap:wrap"></div>
        </div>
      </div>
      <div class="fld">
        <label>Contact name</label>
        <input type="text" id="n-${id}" placeholder="auto-detected after Browse" value="${ea(d.contact_name||'')}">
      </div>
      <div class="fld">
        <label>Relationship</label>
        <input type="text" id="r-${id}" placeholder="Mother" value="${ea(d.contact_relationship||'')}">
      </div>
      <div class="fld">
        <label>Tags <span style="font-weight:400;color:var(--muted)">(comma-separated)</span></label>
        <input type="text" id="t-${id}" placeholder="family, primary" value="${ea((d.tags||[]).join(', '))}">
      </div>
      <div class="fld">
        <label>Framework file <span style="font-weight:400;color:var(--muted)">(optional)</span></label>
        <div class="file-row">
          <input type="text" id="w-${id}" placeholder="optional .md or .txt" readonly value="${ea(d.framework||'')}">
          <button class="btn btn-ghost" onclick="pick(${id},'fw')">Browse…</button>
        </div>
      </div>
      <div class="fld">
        <label>Date format <span style="font-weight:400;color:var(--muted)">(optional — leave blank for auto-detect)</span></label>
        <input type="text" id="hf-${id}" placeholder="e.g. [%d/%m/%y, %H:%M:%S] %name: %text" value="${ea(d.hformat||'')}">
      </div>
    </div>
  </div>`;
}

function addChat(d) {
  const id = _chatId++;
  const wrap = document.createElement('div');
  wrap.innerHTML = chatHTML(id, d);
  document.getElementById('chatList').appendChild(wrap.firstElementChild);
}

function rmChat(id) { document.getElementById(`cr-${id}`)?.remove(); }

function showSenderHint(chatId, senders) {
  const hint = document.getElementById(`sh-${chatId}`);
  const btns = document.getElementById(`sb-${chatId}`);
  if (!hint || !btns) return;
  btns.innerHTML = '';
  senders.slice(0, 4).forEach(name => {
    const btn = document.createElement('button');
    btn.className = 'btn btn-ghost';
    btn.style.cssText = 'font-size:13px;padding:6px 14px';
    btn.textContent = name + ' — this is me';
    btn.onclick = () => {
      document.getElementById('primaryUser').value = name;
      const other = senders.find(s => s !== name) || '';
      const nameField = document.getElementById(`n-${chatId}`);
      if (!nameField.value) nameField.value = other;
      hint.style.display = 'none';
    };
    btns.appendChild(btn);
  });
  hint.style.display = 'block';
}

async function pick(id, type) {
  try {
    if (type === 'chat') {
      const path = await pywebview.api.pick_chat_file();
      if (!path) return;
      document.getElementById(`f-${id}`).value = path;

      // Auto-detect sender names from the file
      const senders = await pywebview.api.detect_senders(path);
      const hint = document.getElementById(`sh-${id}`);
      const btns = document.getElementById(`sb-${id}`);
      if (!senders || senders.length === 0) {
        if (hint && btns) {
          document.getElementById(`sh-lbl-${id}`).textContent = 'No messages found in this file';
          btns.innerHTML = "<span style='color:#ef4444'>Make sure it's a WhatsApp export (.txt) and not a screenshot or PDF.</span>";
          hint.style.display = 'block';
        }
      } else {
        const primaryUser = document.getElementById('primaryUser').value.trim();
        const nameField   = document.getElementById(`n-${id}`);
        if (primaryUser && senders.length >= 2) {
          const isMe = s => s.toLowerCase() === primaryUser.toLowerCase()
                         || s.toLowerCase().includes(primaryUser.split(' ')[0].toLowerCase())
                         || primaryUser.toLowerCase().includes(s.split(' ')[0].toLowerCase());
          const contact = senders.find(s => !isMe(s)) || senders[0];
          if (!nameField.value) nameField.value = contact;
        } else {
          showSenderHint(id, senders);
        }
      }
    } else {
      const path = await pywebview.api.pick_framework_file();
      if (path) document.getElementById(`w-${id}`).value = path;
    }
  } catch(e) { console.error('pick error:', e); }
}

// ── Build config from form ────────────────────────────────────────────────────
function buildConfig() {
  const chats = [];
  document.querySelectorAll('.chat-row').forEach(row => {
    const id = row.id.replace('cr-','');
    const file = document.getElementById(`f-${id}`)?.value||'';
    const name = document.getElementById(`n-${id}`)?.value||'';
    if (!file||!name) return;
    chats.push({
      file,
      contact_name: name,
      contact_relationship: document.getElementById(`r-${id}`)?.value||'',
      tags: (document.getElementById(`t-${id}`)?.value||'').split(',').map(x=>x.trim()).filter(Boolean),
      framework: document.getElementById(`w-${id}`)?.value||null,
      hformat: document.getElementById(`hf-${id}`)?.value||null,
    });
  });
  return {
    primary_user:  document.getElementById('primaryUser').value.trim(),
    chats,
    nlp_engine:    'transformers',
    ollama_url:    document.getElementById('ollamaUrl').value||'http://localhost:11434',
    ollama_model:  document.getElementById('ollamaModel').value||'gemma4',
    no_ai:         document.getElementById('noAi').checked,
    claude_crisis: document.getElementById('claudeCrisis').checked,
    api_key:       document.getElementById('apiKey').value||'',
    custom_topics: [],
  };
}

// ── Run ───────────────────────────────────────────────────────────────────────
async function run() {
  document.getElementById('errBox').style.display = 'none';
  const cfg = buildConfig();
  if (!cfg.primary_user) { showErr('Enter your name in Step 1.'); return; }
  if (!cfg.chats.length) { showErr('Add at least one chat file with a contact name.'); return; }

  setRunning(true);
  document.getElementById('logWrap').style.display = 'block';
  document.getElementById('log').innerHTML = '';

  await pywebview.api.save_config(cfg);
  await pywebview.api.run_analysis(cfg);
}

function setRunning(on) {
  const b = document.getElementById('runBtn');
  b.disabled = on;
  b.textContent = on ? '⏳ Analysing…' : '▶ Run Analysis';
}

function appendLog(msg) {
  const log = document.getElementById('log');
  const d = document.createElement('div');
  d.className = msg.includes('⚠')?'log-warn': msg.includes('✓')||msg.includes('Done')?'log-ok':'';
  d.textContent = msg;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function setError(msg)  { showErr(msg); setRunning(false); }
function showErr(msg)   { const e=document.getElementById('errBox'); e.textContent='⚠ '+msg; e.style.display='block'; e.scrollIntoView({behavior:'smooth'}); }
function toggleKey()    { document.getElementById('apiField').style.display = document.getElementById('claudeCrisis').checked?'block':'none'; }

// ── Updates ───────────────────────────────────────────────────────────────────
async function checkForUpdates() {
  document.getElementById('updateStatus').textContent = 'Checking…';
  document.getElementById('doUpdateBtn').style.display = 'none';
  try {
    const result = await pywebview.api.check_for_updates();
    if (result.error) {
      document.getElementById('updateStatus').textContent = '⚠ ' + result.error;
    } else if (result.available) {
      document.getElementById('updateStatus').innerHTML =
        `<span style="color:#059669;font-weight:600">● Update available</span> — ${result.count} new commit${result.count!==1?'s':''}<br>` +
        `<span style="font-family:monospace;font-size:11px;color:var(--muted)">${(result.preview||'').replace(/</g,'&lt;')}</span>`;
      document.getElementById('doUpdateBtn').style.display = 'inline-flex';
    } else {
      document.getElementById('updateStatus').textContent = '✓ ' + result.message;
    }
  } catch(e) {
    document.getElementById('updateStatus').textContent = 'Error: ' + e;
  }
}

async function doUpdate() {
  document.getElementById('doUpdateBtn').style.display = 'none';
  document.getElementById('updateLogWrap').style.display = 'block';
  document.getElementById('updateLog').innerHTML = '';
  document.getElementById('updateStatus').textContent = 'Updating…';
  await pywebview.api.do_update();
}

function updateLog(msg) {
  const log = document.getElementById('updateLog');
  const d = document.createElement('div');
  d.style.color = msg.includes('✓') ? '#4ade80' : msg.includes('ERROR') || msg.includes('failed') ? '#f87171' : '#94a3b8';
  d.textContent = msg;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function updateDone() {
  document.getElementById('updateStatus').innerHTML = '<span style="color:#059669;font-weight:600">✓ Ready to relaunch</span>';
  document.getElementById('relaunchBtn').style.display = 'inline-flex';
}

async function relaunch() {
  document.getElementById('relaunchBtn').textContent = 'Relaunching…';
  await pywebview.api.relaunch();
}

// ── Ollama model detection ────────────────────────────────────────────────────
let _modelRefreshTimer = null;
function scheduleModelRefresh() {
  clearTimeout(_modelRefreshTimer);
  _modelRefreshTimer = setTimeout(refreshOllamaModels, 800);
}

async function refreshOllamaModels(savedModel) {
  const url = document.getElementById('ollamaUrl').value.trim();
  const btn = document.getElementById('ollamaRefreshBtn');
  const hint = document.getElementById('ollamaHint');
  const sel = document.getElementById('ollamaModel');
  btn.disabled = true;
  btn.textContent = '⟳ Detecting…';
  hint.textContent = '';
  try {
    const models = await pywebview.api.get_ollama_models(url);
    sel.innerHTML = '';
    if (!models || !models.length) {
      sel.innerHTML = '<option value="">— no models found —</option>';
      hint.style.color = '#ef4444';
      hint.textContent = 'Ollama is not reachable or has no models installed.';
    } else {
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        if (m === savedModel || (!savedModel && models.indexOf(m) === 0)) opt.selected = true;
        sel.appendChild(opt);
      });
      hint.style.color = '#059669';
      hint.textContent = `${models.length} model${models.length>1?'s':''} found.`;
    }
  } catch(e) {
    sel.innerHTML = '<option value="">— detection failed —</option>';
    hint.style.color = '#ef4444';
    hint.textContent = 'Could not reach Ollama.';
  }
  btn.disabled = false;
  btn.textContent = '⟳ Detect models';
}

// ── Restore saved config on launch ───────────────────────────────────────────
window.addEventListener('pywebviewready', async () => {
  try {
    const v = await pywebview.api.get_version();
    document.getElementById('versionLabel').textContent =
      v.hash !== 'unknown' ? `${v.hash} · ${v.date}` : 'unknown';
  } catch(e) {}
  try {
    const c = await pywebview.api.load_config();
    if (!c) { addChat(); refreshOllamaModels(); return; }
    if (c.primary_user) document.getElementById('primaryUser').value = c.primary_user;
    if (c.ollama_url)   document.getElementById('ollamaUrl').value   = c.ollama_url;
    if (c.no_ai)        document.getElementById('noAi').checked      = true;
    if (c.api_key)      document.getElementById('apiKey').value      = c.api_key;
    (c.chats||[]).length ? c.chats.forEach(addChat) : addChat();
    // Auto-detect models and select the saved one
    await refreshOllamaModels(c.ollama_model || '');
  } catch(e) { addChat(); refreshOllamaModels(); }
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api = AnalyserAPI()
    window = webview.create_window(
        "WhatsApp Chat Analyser",
        html=CONFIG_HTML,
        js_api=api,
        width=820,
        height=800,
        min_size=(600, 500),
        background_color="#f0f4f8",
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
