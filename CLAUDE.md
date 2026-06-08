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

## Testing Standards

- **Never fix a test by weakening it.** If a test fails, diagnose the root cause first. The failure is either a genuine bug in the code (fix the code) or a genuinely incorrect test assertion (fix the test). Changing the test input or assertion to avoid the failure without understanding why it failed is not acceptable.
- **Tests must reflect real-world expectations.** A test for distress signal detection should use phrases that a real user would actually write. If the phrase is reasonable and the code doesn't catch it, the code is wrong — add or fix the pattern, don't swap in an easier phrase.
- **Understand before you change.** Before modifying any test, be able to explain: (a) why the current test fails, (b) whether the failure reveals a code bug or an incorrect assertion, and (c) what the correct fix is. If the answer to (b) is "the code is wrong", fix the code.
- **New patterns need new tests.** When adding keywords, regex patterns, or detection logic, add a test that would have caught the gap. Don't just make existing tests pass — prove the new behaviour works.
