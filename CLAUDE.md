# WhatsApp Behavioral Analytics App - Guidelines

## Architecture & System Specs

- **Source of Truth:** Refer to `SPEC.md` for the core architecture, regex parser logic, affect aggregator, and local Gemma/Ollama pipeline.
- **Execution:** Read `SPEC.md` before implementing new features or making database/parser edits.
- **Strict Privacy Constraint:** Always ensure model dependencies (e.g., Transformers, SpaCy, Ollama) run strictly locally. Do not introduce remote cloud APIs.

## Local Build & Test Commands

Include commands Claude can use to verify its work:

- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest`
- Linting: `flake8`

## Coding Standards

- Keep code reviewable: write small, focused diffs.
- Ensure proper lookaheads are utilized in regex patterns to prevent multi-line message truncation.
- After implementing features, update `SPEC.md` if any route schemas or database logic changes.
