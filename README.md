# Healthcare Triage Assistant

**An AI-primary, human-in-the-loop, multimodal triage-support tool for government hospitals, PHCs,
public health camps, company clinics, industrial-estate health units, and campus health centers -
built to run fully offline on CPU-only hardware, and hardened for real deployment, not just a
five-minute demo.**

> ⚠️ **This is an educational hackathon project.** It is **non-diagnostic** — it organizes
> information and highlights urgency signals for a qualified reviewer, but it never prescribes
> treatment and never replaces a health worker, nurse, doctor, or medical officer. Do **not**
> enter real patient data; only synthetic/demo data is used or should be used with this build.

---

## 0. What "industrial-grade" means in this codebase

A hackathon prototype and a system someone could actually run in a PHC differ in specific,
checkable ways. This project was hardened along exactly those lines:

| Concern | What changed |
|---|---|
| **AI reliability** | Extraction is grammar-constrained (GBNF, from a JSON Schema) so the model's output is forced to be valid JSON at the token-sampling level, not just "asked nicely" in the prompt - this is the single biggest lever on "every case gets solved by the AI model" instead of silently falling back. One automatic retry backs up extraction, summary, and translation before falling back to the deterministic path. See §3. |
| **CPU performance** | Thread count, batch size, and context size are tuned and environment-overridable; the model is preloaded and warmed up on a background thread at process start instead of stalling the first request; every inference call is timed and exposed via `/health`. See §3.4. |
| **Security** | Hashed passwords (with transparent migration from the old demo's plaintext rows), a real persistent secret key, hardened session cookies, CSRF protection on every state-changing form, security headers, and a login rate limiter. See §6. |
| **Reliability / ops** | Structured, rotating file logging; every uncaught exception is caught, logged with a traceback, and shown as a friendly error page instead of a stack trace; database writes run inside a proper commit/rollback transaction boundary; server-side input validation on every form. See §7. |
| **Deployability** | A production WSGI entrypoint (`wsgi.py`) + `gunicorn.conf.py`, a `Dockerfile` + `docker-compose.yml`, environment-driven configuration (`config.py`, `.env.example`) instead of hardcoded values. See §8. |
| **Verifiability** | A 181-test automated suite (`tests/`, runnable with plain `unittest` or `pytest`, zero extra setup) covering the safety-critical merge logic, the AI safety filter, security primitives, vision/translation model gating, translation phrase-bank completeness, PDF export, and full HTTP request/response behavior with an isolated database. See §9. |

Nothing about the actual triage/safety design changed - §§1-5 below describe the same
AI+rules hybrid pipeline as before; §§6-9 describe what's new.

---

## 1. Quick start

Requirements: Python 3.9+, `tesseract-ocr` and `poppler-utils` installed at the OS level (already
present on most Debian/Ubuntu systems via `apt install tesseract-ocr poppler-utils`).

```bash
cd triage-assistant
pip install -r requirements.txt

python seed.py       # one-time: creates the DB, demo users, and synthetic demo cases
python app.py        # starts the server on http://localhost:5000
```

Or simply run `./run.sh`, which seeds automatically on first run.

Copy `.env.example` to `.env` if you want to override any default (secret key, log level, AI
thread count, rate limits, etc.) - every setting also works as a real environment variable, so
`.env` is a convenience, not a requirement (see §6.2 and §8).

### Enabling the local AI model (optional, one-time download)

The app works fully out of the box with a deterministic, rules-based extraction pipeline. To
additionally enable the **local AI model** (offline, CPU-only, no API key, no internet needed
after this one download) that becomes the **primary symptom-extraction and summarization engine**:

```bash
python download_model.py     # one-time, ~1 GB, needs internet
python app.py                # AI extraction activates automatically once the file is present
```

This downloads `Qwen2.5-1.5B-Instruct` (quantized to GGUF) via `llama-cpp-python`. After this one
download, **everything runs fully offline** — no API keys, no outbound network calls of any kind
at request time (verified: see §5.1). If the model file isn't present (or `llama-cpp-python`
isn't installed), the app automatically falls back to the deterministic rules-only pipeline with a
clear on-screen note — nothing breaks either way.

An **experimental local vision model** (moondream2, also via `llama-cpp-python`) can additionally
be downloaded to generate a descriptive (non-diagnostic) caption of uploaded photos:

```bash
python download_model.py --vision     # downloads both text + vision models
python download_model.py --vision-only
```

See §11.5 for an important caveat: this code path could not be exercised end-to-end in the build
environment (no internet access there) and should be tested on a machine with internet access
before relying on it in a demo.

A separate, **medical-domain vision model** (Google's MedGemma-4B-IT) can additionally be
downloaded to generate an **AI-suggested, hedged, clinician-facing differential** of possible
visible conditions from a photo — not just a neutral description:

```bash
python download_model.py --medgemma     # ~4.2 GB, gated - see below if it 401s/403s
```

MedGemma sits under Google's "Health AI Developer Foundations" terms of use, so unlike every
other model this project downloads anonymously, the download above may need a Hugging Face
token: accept the terms at the printed URL, create a token, then re-run with
`HF_TOKEN=hf_xxx python download_model.py --medgemma`. See §11.4 for the full design of this
feature — in particular, why it's the one deliberate, narrow exception to this project's
"AI never names a condition" rule everywhere else, and exactly how it is kept safe anyway.

Qwen2.5 (the default text model above) has **weak, unreliable Odia coverage** — Odia isn't one of
its officially supported languages, which is exactly why this project already had to build
script-mismatch and repetition-loop detection around its translations (§11.6). A dedicated
**English→Indic machine-translation model** (IndicTrans2, from AI4Bharat) can be downloaded to
replace it for this one task:

```bash
pip install transformers torch IndicTransToolkit   # extra packages, only needed for this feature
python download_model.py --indic-translate          # ~1 GB - see note below, this WILL fail without a token
```

This model's repo is **gated on Hugging Face** (despite being MIT licensed) - if the download
above fails, accept the terms at
https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M, create a token at
https://huggingface.co/settings/tokens, then re-run with
`HF_TOKEN=hf_xxx python download_model.py --indic-translate` (same flow as MedGemma above). Without
it, translation silently runs on the weaker Qwen2.5 fallback instead - see §11.6's "real, confirmed
gap" note for exactly why that looks like poor Hindi/Odia quality rather than a missing model.

See §11.6 for why a dedicated MT model beats asking a generalist chat model to translate into a
low-resource script, and the full fallback chain if it isn't installed (IndicTrans2 → Qwen2.5 →
English, never a crash either way).

### Demo accounts (all synthetic, password `demo123`)

| Username    | Role                          | What they can do                                   |
|-------------|-------------------------------|-----------------------------------------------------|
| `worker1`   | Health Worker (intake)        | Record consent + patient-reported symptoms/reports  |
| `reviewer1` | Medical Officer (reviewer)    | Dashboard, triage detail, sign-off, referral export  |
| `auditor1`  | Facility Data Officer         | Audit log, data-minimisation ("purge closed") control|

Passwords are stored **hashed** (see §6.1), including these demo accounts - `seed.py` hashes them
before insert. If you have an existing database from before this security pass, the first
successful login for each user transparently upgrades that row to a real hash; nothing needs to
be done manually.

This role split is a deliberate, minimal mock-up of **role-based access control** — it shows the
intended architecture without building a full identity/security stack (SSO, MFA), which is out of
scope for a project at this stage (see §10 for what a real deployment would add on top).

---

## 2. What it does (mapped to the problem statement's evaluation criteria)

| Criterion | Weight | How this project addresses it |
|---|---|---|
| **Safety-first triage workflow** | 20% | Risk-tiering (Emergency / High / Medium / Low) is computed by a deterministic, auditable rules engine (`triage_engine/risk_rules.py`) — this is a **safety decision, not a technical limitation** (see §5.1). The AI model is the **primary** engine for reading the patient's text (§5.2), but it is architecturally unable to *lower* urgency — `triage_engine/pipeline.py` runs the rules engine on the union of what the rules **and** the AI each independently detected, so the AI can only add urgency signals the keyword engine missed, never silently remove one it caught. This exact guarantee has a dedicated automated regression test (`tests/test_pipeline.py::test_ai_cannot_downgrade_a_rules_detected_red_flag`). The reviewer can see both numbers side by side (an expandable "rules-only cross-check" panel on every AI-powered note). |
| **Quality of information extraction & summarization** | 20% | The local AI model (`triage_engine/llm.py` + `pipeline.py`) is the primary path: it reads the raw patient text and returns a structured JSON extraction (chief complaint, symptoms with severity/category, negated symptoms, timeline, missing info, follow-up questions). As of the industrial-grade pass, this JSON is **grammar-constrained** (see §3) so it reliably parses, plus one automatic retry, so the overwhelming majority of cases are genuinely solved by the model rather than silently falling back. If the model is unavailable or its output still fails validation after retrying, the system **transparently falls back** to the original deterministic keyword/regex extractor (`extractor.py`) — every note is labeled which path produced it ("🤖 AI-Extracted" vs "📋 Rules-Based"), and `logs/app.log` records exactly why on every fallback. |
| **Multimodal capability** | 15% | **Text** intake, **voice** intake (browser-native Web Speech API, no server dependency, follows the same English/Hindi/Odia "Preferred language" selector rather than being locked to English - see `static/js/intake.js`'s honesty note on real-world browser support varying by language), **OCR** for uploaded lab-report/vitals images or PDFs (`triage_engine/ocr.py`), a general-purpose local vision model (moondream2, the default, fast, CPU-friendly choice — see §11.5) for a plain-language, non-diagnostic description of an uploaded photo, and an opt-in **MedGemma-4B** (`VISION_PREFER_MEDGEMMA=1`), a medical-domain vision model that produces an AI-suggested, hedged differential of possible visible conditions from the same photo, for clinician confirmation, at the cost of a much slower per-photo response on CPU-only hardware (`triage_engine/vision.py` - see §11.4 for the safety design and the speed/memory tradeoff, §11.5 for moondream2's untested status). |
| **India-wide facility relevance & accessibility** | 15% | Facility-type selector spanning all six facility categories named in the problem statement; a scenario picker covering all seven suggested India-wide scenarios, including **maternal-health follow-up** and **chronic-disease check-in**, driving a real reminder/tracking feature (§5.3); a full English/Hindi/**Odia** UI and note-language toggle, with the AI summary translated by a dedicated English->Indic MT model (IndicTrans2, preferred over the generalist chat model specifically for Odia's weak coverage there) and quality-checked before display either way (script + repetition detection, §11.6); lightweight server-rendered pages (no heavy JS framework, no CDN dependency - see §6.4's CSP note) for low-bandwidth/low-end-device use; voice input for lower-literacy users, itself available in English/Hindi/Odia (matching the patient's own preferred language rather than English-only) so a patient from a different linguistic background can speak symptoms in their own words. |
| **Human-review design & escalation logic** | 15% | A sidebar-navigation reviewer dashboard with stat tiles, a dedicated "Upcoming Follow-ups & Check-ins" panel with overdue flagging, queues notes by priority tier then age, with filters by status/facility/tier. Every note requires an explicit reviewer action before being considered handled; a one-click **referral PDF** export supports handoff to a higher facility. |
| **Privacy & responsible AI controls** | 10% | Mandatory **consent** screen before every intake; minimal data collection (name is optional); patients are referenced everywhere by a generated **Triage ID**, not name; the reviewer view **masks patient names**; every view/edit/export is written to an **audit log**; an auditor-only **data-minimisation control** purges closed records; every page carries a non-diagnostic disclaimer banner; passwords hashed, sessions hardened, CSRF-protected forms, security headers (§6); AI-specific safeguards described in §5.2 and §6. |
| **Demo Quality** | 5% | `seed.py` pre-populates realistic synthetic cases across all four risk tiers, several facility types, OCR'd sample lab slips, and follow-up reminders. `/health` gives a judge or operator a live, structured view of database and AI-model status (including real latency numbers, §3.4) rather than just a claim. |

---

## 3. AI reliability & CPU optimization (industrial-grade pass)

`triage_engine/llm.py` runs a small local model (Qwen2.5-1.5B-Instruct, quantized to GGUF Q4_K_M,
~1GB) via `llama-cpp-python`, entirely offline and CPU-only. The hardening work here targeted one
concrete goal: **make "every diagnosis-support case is solved by the AI model" actually true in
practice**, not just true when the input happens to be easy for a 1.5B model.

### 3.1 Grammar-constrained JSON extraction

The old approach asked the model nicely, in the prompt, to output JSON matching a schema - and
fell back to the rules engine whenever a small model added a stray markdown fence, trailing prose,
or a slightly malformed value (this was, empirically, the single biggest real cause of fallback).

The new approach builds a JSON Schema (`llm.EXTRACTION_JSON_SCHEMA`) and compiles it into a GBNF
grammar via `llama_cpp.LlamaGrammar.from_json_schema(...)`, then passes that grammar to
`create_chat_completion(..., grammar=...)`. This restricts *at the token-sampling level* which
next tokens are even legal, so the model cannot produce a token sequence that violates the
schema's shape (wrong types, extra top-level prose, a markdown fence, an invalid enum value).
This is also typically **faster**, not slower, since it prunes the sampling space instead of
letting the model wander into an invalid sequence that then has to be discarded.

If the installed `llama-cpp-python` version doesn't support grammar-from-schema, this degrades
gracefully to the old prompt-only behavior (logged once at startup) rather than crashing.

**A real bug found on first actual-hardware run, and its fix**: grammar constrains *shape*, not
*length* - a verbose case (e.g. a long missing_info list on a pediatric case) could still exhaust
`max_tokens` before the JSON's closing braces were generated, leaving a syntactically incomplete
document that fails to parse, deterministically, on every retry (low-temperature decoding
reproduces the same truncation point). The schema now caps every array (`maxItems`) and string
(`maxLength`) so a worst-case response fits comfortably in budget (see `MAX_SYMPTOMS`,
`MAX_CHIEF_COMPLAINT_LEN`, etc. in `llm.py`), the prompt itself asks for conciseness, the base
token budget was raised, and - crucially - **each retry uses a wider token budget than the last**
(`EXTRACTION_MAX_TOKENS` → `EXTRACTION_MAX_TOKENS_RETRY`) instead of repeating the same budget and
reproducing the identical failure. `N_CTX` was also raised (2048 → 4096) to give the wider retry
budget room alongside a full-length prompt. Locked in by
`tests/test_llm_safety.py::TestExtractionRetryWidensTokenBudget`.

### 3.2 Automatic retry before falling back

Extraction, summary generation, and translation each get one automatic retry
(`LLM_MAX_RETRIES`, default 1) before the deterministic fallback kicks in, since a small local
model occasionally produces an empty or borderline response on a single attempt but succeeds
immediately on a second try. Every retry and every eventual fallback is logged with the reason.

### 3.3 CPU thread/batch/context tuning

`N_THREADS`, `N_BATCH`, and `N_CTX` (env-overridable: `LLM_N_THREADS`, `LLM_N_BATCH`,
`LLM_N_CTX`) are passed straight to `llama_cpp.Llama(...)`, along with `use_mmap=True` (fast load,
OS page-cache-friendly across restarts) and `use_mlock=False` (avoids requiring elevated
memory-lock privileges in containers). Defaults are tuned for a typical multi-core CPU-only
deployment (`n_threads = CPU count - 1`) but are meant to be re-tuned per actual deployment
hardware without touching code - see `.env.example`.

### 3.4 Preload, warm-up, and real performance numbers

The model is loaded **and given one throwaway warm-up generation** on a background thread as soon
as the Flask app module is imported (`llm.preload_async()`, called from `app.py` at import time -
not gated behind `if __name__ == "__main__":`, so it also fires correctly under gunicorn). This
means the first real user request doesn't pay the full ~1GB load time.

Every model call is timed. `GET /health` exposes the live numbers:

```json
"ai_text_model": {
  "available": true,
  "stats": {
    "model_loaded": true, "load_seconds": 4.8,
    "n_threads": 7, "n_batch": 512, "n_ctx": 2048,
    "calls": 142, "failures": 3,
    "avg_latency_ms": 890.4, "last_latency_ms": 812.1
  }
}
```

This turns "optimized CPU inference" into something a judge, or an ops dashboard, can see numbers
for on the actual demo machine - not just a claim in this document.

### 3.5 What stays exactly as it was, on purpose

The AI model still never decides risk tier (§5.1) and every output still passes the same
diagnosis/prescription/dosage safety filter (§6.3-equivalent content, unchanged) before display.
Grammar-constraining the *shape* of extraction output does not relax what *content* is allowed
through - both layers apply independently.

---

## 4. Suggested demo script (~5–6 minutes)

1. **Sign in as `worker1`** → consent screen → intake form. Type (or use the 🎤 voice button for)
   a symptom description, optionally attach one of the generated sample lab slips from
   `sample_data/`, and submit. Point out the instant confirmation screen showing the generated
   Triage ID, priority tier, and *why*.
2. **Sign in as `reviewer1`** → Dashboard. Point out the priority stat tiles, the **Upcoming
   Follow-ups & Check-ins** panel (with an OVERDUE badge on one demo case), and that the main queue
   is sorted emergency-first regardless of arrival order.
3. Open the **emergency** case ("Chest pain and breathlessness"). Point out the **🤖 AI-Extracted /
   📋 Rules-Based** badge next to the priority badge, the AI-generated narrative summary (if
   `download_model.py` has been run), and the expandable **"Rules-based safety-net cross-check"**
   panel showing that the deterministic engine independently agrees with (or would have produced a
   lower tier than) the final priority — demonstrating the AI can escalate but never downgrade.
   Switch the language toggle through English → Hindi → **Odia** to show the multilingual
   rendering.
4. Open the case with an attached lab slip and show the **OCR-extracted vitals** feeding directly
   into the risk score (a low SpO2 reading on the slip escalates the case even if the text alone
   wouldn't).
5. Open the maternal-follow-up or chronic-check-in demo case, show the **next follow-up date**
   field, then go back to the dashboard and click **"Mark follow-up complete"** to show the
   reminder disappearing from the panel.
6. Change a note's status, add a reviewer note, and click **Download Referral PDF**.
7. **Sign in as `auditor1`** → Audit Log, to show every action so far was recorded, then run
   **Purge Closed Records** to show the data-minimisation control.
8. **If a judge asks about production-readiness**: open `GET /health` in a new tab to show live
   AI-model performance stats, mention hashed passwords / CSRF / rate limiting (§6), and point at
   `tests/` (70 automated tests, runs in ~3 seconds, `python -m unittest discover -s tests`).

---

## 5. Architecture

```
app.py                     Flask routes / controllers, security wiring, error handling
config.py                  Environment-driven configuration (dev/prod/testing profiles)
wsgi.py                    Production WSGI entrypoint (gunicorn/uwsgi import this)
gunicorn.conf.py           Production WSGI server tuning (see §8.2)
Dockerfile, docker-compose.yml, docker-entrypoint.sh   Containerized deployment (see §8.3)
triage_engine/
  database.py              SQLite schema + connection helpers (WAL mode, context-manager
                            transactions, health check) - see §7.3
  security.py               CSRF protection, login rate limiter, password hashing +
                            transparent legacy-plaintext migration - see §6
  risk_rules.py             Symptom ontology, negation handling, vitals thresholds, risk scoring
  extractor.py              Free-text -> structured note (deterministic fallback path)
  pipeline.py               Orchestrator: runs AI extraction when available, unions its findings
                             with the rules engine's, and always computes the final risk tier via
                             the deterministic engine (see §5.1-§5.2) - THIS is what "the problem
                             statement is solved by an AI model" means architecturally
  llm.py                    Local, offline, CPU-only text model (Qwen2.5-1.5B-Instruct via
                             llama-cpp-python): grammar-constrained structured extraction,
                             narrative summary, extra follow-ups, AI translation - with
                             prompt-injection defenses, output-side safety filtering, retries,
                             and performance instrumentation (see §3)
  vision.py                 Local, offline, CPU-only vision models via llama-cpp-python: moondream2
                             as the default, fast, general-purpose descriptive model (experimental,
                             see §11.5), MedGemma-4B as an opt-in (VISION_PREFER_MEDGEMMA=1) model
                             for AI-suggested, hedged, clinician-facing possible findings from a
                             photo, at the cost of much slower CPU inference (see §11.4)
  ocr.py                    Image/PDF -> text (pytesseract + pdf2image) + basic image-quality check
                             (+ is_usable_ocr_text() guard - see note right after this tree, below,
                             for a real routing bug this fixes)
  translator.py             EN/HI/OR phrase-bank translation for UI strings & fixed follow-up
                             questions (documented scope, see §5.5)
  indic_translate.py        Dedicated English->Hindi/Odia MT (IndicTrans2 via transformers/torch),
                             preferred over Qwen2.5 for the AI summary translation (see §11.6)
  summarizer.py              Structured note -> plain-text rendering (reviewer view / referral)
  referral.py                Structured note -> referral PDF (reportlab)
  audit.py                  Audit logging + name-masking helpers
templates/, static/         Server-rendered UI (Jinja2 + vanilla JS + hand-rolled CSS design
                             system, no build step, no CDN dependency - see §5.6)
tests/                       181-test automated suite - see §9
seed.py                     Seeds demo users (hashed passwords) + synthetic cases
sample_data/                Synthetic sample images used for the OCR demo
download_model.py           One-time download of the local AI text model (and optionally the
                             vision model) - see §1
models/                     Downloaded .gguf model file(s) live here (empty until download_model.py runs)
data/                       Runtime SQLite DB, auto-generated secret key, uploaded report files
logs/                       Rotating application log files (see §7.1)
```

No network calls are made at runtime — OCR, risk scoring, extraction, AI extraction/summarization,
and (if downloaded) vision description all run locally on CPU, so the demo works offline and
reproducibly. The only network use in the whole project is the one-time model download(s).

**How an upload is routed by file type, and a real bug found in that routing**: there is one
upload field (`report_file`), used for both a photographed symptom (skin/wound/eye/swelling, or an
X-ray) and a photographed/scanned lab report or prescription - the app can't know in advance which
one a given file is, so `app.py`'s intake handler always runs OCR on every upload, and additionally
runs the **vision model** (moondream2, or MedGemma if `VISION_PREFER_MEDGEMMA=1`) on every upload
that isn't a PDF. For a PDF, only OCR runs (the vision model can't read PDFs) and its extracted
text feeds `llm.generate_ai_extraction()` (the text model) - exactly "PDF prescription -> text
model" as intended. For an image, both run - but `ocr.run_ocr()` correctly finds no usable text on
a genuine symptom photo or X-ray and returns a placeholder message
(`"[No text could be extracted from this file...]"`) for the reviewer to read.

That placeholder string was a real, confirmed bug here: it was being fed straight into both the
rules-based vitals scan and the AI extraction prompt as if it were real report content (any
non-empty string counted as "OCR found something"), which meant an image of a genuine symptom or
an X-ray could have its extraction quietly driven by that placeholder message instead of by the
vision model's actual description - looking, from the outside, like "OCR is doing the diagnosis"
instead of moondream2. Fixed by `ocr.is_usable_ocr_text()`, which recognizes OCR's own
placeholder/error strings and excludes them before `pipeline.run_triage_pipeline()` passes
`ocr_text` down to either extraction path - the reviewer still sees the original placeholder
message for their own information, but it never reaches the model as if it were content. See
`tests/test_ocr.py` and `tests/test_pipeline.py::TestOcrPlaceholderTextIsFiltered`.

**One remaining, deliberate limitation, not a bug**: neither vision model is trained on
radiology. moondream2's prompt is instructed to describe visible characteristics only, never to
name a condition (see `vision.py`'s `DESCRIBE_PROMPT`) - for an X-ray it will produce a generic
visual description, not a radiological read. MedGemma's hedged "possible findings" feature
(`FINDINGS_PROMPT`) is explicitly scoped to an externally visible symptom ("skin, wound, eye,
swelling, etc.") and will most likely respond "Not applicable" for an actual X-ray rather than
attempt one - this is intentional, since neither model has real radiology training and a wrong
hedge on an X-ray is a meaningfully worse failure mode than a wrong hedge on a visible skin
condition. If genuine (still non-diagnostic, hedged) radiological commentary is wanted, that's a
deliberate, separate feature addition, not something either model does out of the box.

---

## 6. Security model

### 6.1 Passwords

Stored as salted hashes (`werkzeug.security.generate_password_hash`, PBKDF2/scrypt depending on
platform support) - never plaintext, including the synthetic demo accounts. `seed.py` hashes on
insert. `triage_engine/security.verify_and_upgrade_password()` supports a database seeded before
this security pass (plaintext rows): it verifies against the plaintext value once, then
transparently rewrites that row to a real hash - so upgrading never locks anyone out and never
requires a manual migration step (`tests/test_security.py::TestPasswordMigration` covers both
paths).

### 6.2 Secret key & session cookies

`config.py` generates a strong random key on first run (`secrets.token_hex(32)`) and persists it
to `data/.secret_key` (mode 0600, gitignored) so sessions survive a restart without a secret ever
being committed to source control - or reads `SECRET_KEY` from the environment for a real/
multi-instance deployment. Session cookies are `HttpOnly`, `SameSite=Lax`, have a configurable
lifetime (`SESSION_LIFETIME_MINUTES`, default 60), and `SESSION_COOKIE_SECURE` can be turned on via
environment variable once the app sits behind TLS.

### 6.3 CSRF protection

Every state-changing route (`POST`) requires a per-session token (`triage_engine/security.py`):
issued via a Jinja context processor (`{{ csrf_token() }}`, present as a hidden field on all 7
POST forms in the templates) and checked in a `before_request` hook with a constant-time
comparison. Implemented without adding Flask-WTF as a dependency - deliberately, so the entire
mechanism is under 40 lines a reviewer can read directly rather than trusting a black-box import
(see the module docstring for the reasoning). Covered by both unit tests
(`tests/test_security.py`) and a live end-to-end check (`tests/test_app_integration.py::
test_post_without_csrf_token_is_rejected`, plus a real-browser Playwright pass during development).

### 6.4 Security headers

`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: strict-origin-when-cross-origin`, and a `Content-Security-Policy` that blocks
every external script/style/object/frame origin (the app ships its own CSS/JS with zero CDN
dependency - see §5.6 - so `default-src 'self'` costs nothing functionally). `'unsafe-inline'` is
allowed for style/script specifically because the existing templates use a handful of inline
`style="..."` attributes and one inline `<script>` block (`intake.html`) - tightening this further
to a nonce-based CSP is a natural next step (§10) but was not necessary to get real protection
against the more common external-exfiltration and clickjacking payloads.

### 6.5 Login rate limiting

A dependency-free sliding-window limiter (`triage_engine/security.RateLimiter`) throttles login
attempts per client IP (default: 10 per minute, `RATELIMIT_LOGIN` env var). In-memory and
per-process by design for a single-instance deployment; see §10 for the multi-instance upgrade
path (a shared store such as Redis).

### 6.6 Server-side input validation

Beyond HTML5 form attributes (which any client can bypass), `app.py`'s `_validate_intake_form()`
checks facility type, scenario, sex, age format, symptom-text presence/length, and follow-up-date
validity server-side before anything reaches the pipeline or the database, with the same
constraints re-checked on the reviewer's update form.

### 6.7 What this is *not*

This is still a demo-account model, not a real identity system: no self-service registration, no
MFA, no password reset flow, no per-facility account isolation. That's an intentional scope
boundary for a project at this stage (a full identity/security stack is its own project) - see
§10 for what a real deployment on top of this would need to add.

---

## 7. Reliability & operations

### 7.1 Logging

Every request, every AI decision, every safety-filter rejection, and every fallback is logged
through Python's standard `logging` module to both the console and a rotating file
(`logs/app.log`, 5MB x 5 backups, `LOG_LEVEL` env var). This means "why did this case fall back to
rules?" or "why did this translation get suppressed?" is always answerable after the fact from the
log, not just from re-reading the code.

### 7.2 Error handling

Every route is covered by global error handlers (400/403/404/429/500 + a catch-all for any
unhandled exception) that log a full traceback server-side and show a clean, on-brand error page
to the user - never a raw stack trace. The three most complex write paths (intake creation, AI
regeneration, status update) additionally wrap their pipeline/database calls in explicit
try/except blocks with user-facing, actionable flash messages on failure.

### 7.3 Database transaction safety

`triage_engine/database.session()` is a context manager that commits on clean exit and rolls back
(then re-raises) on any exception, always closing the connection - so a mid-request error can
never leave a half-written transaction or a leaked connection behind. `PRAGMA journal_mode=WAL` is
enabled so the dashboard (a reader) doesn't block on a concurrent intake (a writer). `GET /health`
runs a real `SELECT 1` against the database as a liveness/readiness check, not just a static "ok".

---

## 8. Deployment

### 8.1 Local / development

`python app.py` (Flask's built-in dev server) - fine for iteration and the hackathon demo itself,
not intended for anything beyond that (Flask's own documentation says the same).

### 8.2 Production WSGI

```bash
pip install -r requirements.txt   # includes gunicorn
export APP_ENV=production
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
gunicorn -c gunicorn.conf.py wsgi:app
```

`gunicorn.conf.py` deliberately keeps worker count low (`min(cpu_count, 4)`, override with
`WEB_CONCURRENCY`) and the request timeout high (120s, override with `GUNICORN_TIMEOUT`) because
this app is CPU-bound and each worker process loads its **own** full copy of the local AI model -
see the file's comments for the RAM-planning implication (`workers x ~1.5-2GB`).

### 8.3 Docker

```bash
docker compose up --build
docker compose exec triage-assistant python download_model.py   # one-time, needs internet
```

`data/`, `models/`, and `logs/` are bind-mounted so the database, the auto-generated secret key,
downloaded model weights, and logs all survive a rebuild. The container runs as a non-root user,
exposes `/health` as its Docker `HEALTHCHECK`, and seeds demo data automatically on first boot only
(never overwriting an existing database) via `docker-entrypoint.sh`.

---

## 9. Automated tests

```bash
python -m unittest discover -s tests -v      # zero extra dependencies, ~3 seconds
# or, if you have pytest installed:
pytest
```

181 tests across 10 files (plus `__init__.py`), none of which need the ~1GB+ model weights
downloaded (the AI/vision/translation models are mocked where relevant) and none of which touch
your real `data/triage.db` (the integration suite monkeypatches `database.DB_PATH` to a temp file
before importing the app):

| File | What it covers |
|---|---|
| `test_risk_rules.py` | Symptom/negation detection (including compound "X or Y" negation), vitals extraction (including blood-pressure plausibility bounds), danger-threshold evaluation, combo rules, tier computation, age-unit handling (days/months/years) (25 tests) |
| `test_llm_safety.py` | The safety filter (including a direct regression test for a real bug: the filter once rejected every compliant summary because of its own mandatory disclaimer line), extraction JSON validation, JSON parsing, translation quality checks (25 tests) |
| `test_pipeline.py` | The core safety guarantee - the AI can add symptoms the rules engine missed, but can never cause the final tier to drop below what the rules engine alone would have assigned; the OCR-placeholder-text filter; OCR text now being scanned for red-flag symptoms; the AI-symptom-name ontology matcher no longer over-matching short/generic names to unrelated red flags; negated-symptom canonical mapping (17 tests, AI mocked) |
| `test_security.py` | CSRF token issuance/validation, the rate limiter's sliding window, and both password-verification paths (hashed + legacy-plaintext-with-upgrade) (12 tests) |
| `test_app_integration.py` | Full Flask test-client coverage: login/logout, the open-redirect fix on `next=`, CSRF enforcement, role-based access control on every route (including the intake-worker note-ownership fix), the complete consent→intake→dashboard flow (including sticky form values after a validation error), `/health`'s response shape, on-demand translation of every AI-generated field with caching (including cache invalidation on regenerate), translated/humanized display of denied symptoms and symptom categories, and the triage detail page's static labels/risk-tier badge/status actually rendering in Hindi/Odia (39 tests) |
| `test_ocr.py` | The `is_usable_ocr_text()` filter that distinguishes real OCR output from "no text extracted"/"OCR failed" placeholder strings (6 tests) |
| `test_vision_diagnosis.py` | Vision model preference ordering (moondream2 default, MedGemma opt-in), findings parsing/truncation, the safety filter applied to hedged findings, image downscaling, and graceful degradation without model files (25 tests) |
| `test_indic_translate.py` | Language gating (only hi/or attempt translation), graceful degradation when torch/models are unavailable, and reuse of `llm.py`'s translation quality checks (10 tests) |
| `test_translator.py` | Completeness of the Hindi/Odia phrase banks in `translator.py` - every canonical symptom, symptom category, fixed follow-up question, risk tier, note status, and UI string must have a Hindi and an Odia entry, so a future addition can't silently regress to English-only display (16 tests) |
| `test_referral.py` | Referral PDF generation doesn't crash on unescaped `<`/`&` in reviewer notes, AI summary, chief complaint, or risk reasons (6 tests) |

---

## 10. Known limitations / what a larger production deployment would still add

- A full identity system (self-service accounts, MFA, password reset, per-facility isolation)
  instead of the fixed demo-account model (§6.7).
- A shared rate-limit store (Redis, etc.) instead of the in-memory per-process limiter, if scaling
  beyond one instance/worker process (§6.5).
- A nonce-based CSP instead of `'unsafe-inline'` for script/style, which would require removing the
  handful of remaining inline `style="..."` attributes and the one inline `<script>` block (§6.4).
- A larger/instruction-tuned model (or a hosted API model) if higher-quality AI extraction/summaries
  are needed than a 1.5B CPU model provides — swapping this in only touches `triage_engine/llm.py`.
- A more robust prompt-injection defense than keyword fencing + output filtering (e.g. a dedicated
  guard model, or structured-output constraints) before trusting AI output in a real clinical
  workflow.
- Verified, on-hardware testing of the vision-model integrations (§11.4, §11.5) and IndicTrans2
  (§11.6) with the actual downloaded weights, since the build environment could not reach the
  internet to do this itself.
- Age-banded pediatric/geriatric vital-sign thresholds (currently adult defaults + a simple
  age adjustment).
- A configurable retention policy instead of the manual "purge closed" demo button.
- Integration with a real speech-to-text service for languages the browser's built-in recognizer
  doesn't support well.
- A native-speaker-verified pass over the Odia phrase bank (§5.5), and over IndicTrans2's actual
  Odia output (§11.6), before any real-world use.

---

## 11. Deliberate scope decisions (read this before judging extraction/translation/AI quality)

This was built to run **fully offline and CPU-only**, and to make the AI model the primary engine
for the problem statement while never weakening the safety-first requirement. Each decision below
is documented in its module's docstring too, not hidden.

### 11.1 Why risk-tiering is, and must stay, deterministic rules + regex

`risk_rules.py` computes the final Emergency/High/Medium/Low tier — never the AI model directly.
This is a safety decision, not a technical limitation: a small local LLM can be wrong, can
hallucinate, or can be manipulated by adversarial patient-entered text (prompt injection), and an
auditable, reproducible rule ("SpO2 < 92% ⇒ escalate") is something a reviewer and a regulator can
actually verify after the fact. The problem statement itself lists "rules-based risk flags" and
"lightweight LLM summarization" as two separate recommended technologies — this project keeps
them as two separate, independently-checkable layers, joined by one rule: **the AI can only add
urgency signals, never remove ones the deterministic layer already found.** This is now backed by
an automated regression test, not just a design claim (§9).

Verified offline behavior: during regression testing, the full demo flow (login, intake with an
image upload, AI-summary regeneration attempt, language switching, referral PDF export, audit log,
purge-closed) was driven end-to-end through a headless browser with all outbound network requests
logged. **Zero requests left the machine** — confirming the app makes no network calls of any kind
at runtime, with or without the AI model downloaded.

### 11.2 The AI model as the primary extraction engine (`pipeline.py` + `llm.py`)

1. `pipeline.run_triage_pipeline()` always runs the deterministic extractor first, as a baseline
   and safety net.
2. If the local model is available, it is prompted, with output constrained to a strict JSON
   schema at the grammar level (§3.1), explicit instructions to never diagnose/prescribe, and the
   patient's free text fenced off as untrusted input with instructions to ignore anything that
   looks like an embedded command, to independently extract chief complaint, symptoms, negated
   symptoms, timeline, missing info, and follow-up questions.
3. The AI's output is validated field-by-field (allowed severity/category values, length/item
   caps) and passed through the same output-side keyword safety filter used for the narrative
   summary. Unusable output gets one automatic retry (§3.2), then falls back to the deterministic
   result — the reviewer always sees which path was used.
4. The final structured note **unions** the rules-detected and AI-detected symptoms; the risk tier
   is computed by `risk_rules.compute_risk()` on that union, so the AI genuinely drives what is
   displayed and what drives urgency-scoring, while remaining unable to suppress a red flag the
   regex layer already caught.
5. Every AI-powered note stores and displays what the rules-only pass alone would have produced
   (`rules_only_risk_tier`, `rules_only_symptoms`), in a collapsible "safety-net cross-check" panel
   — this is deliberately visible, not hidden telemetry, so a reviewer or judge can verify the claim
   in (4) directly on real output.

### 11.3 Follow-up reminders are a working feature, not just a scenario label

`maternal_followup` and `chronic_checkin` scenarios capture a `next_followup_date` at intake (or
later, from the reviewer view), surfaced on the dashboard as an **Upcoming Follow-ups & Check-ins**
panel with automatic overdue detection, and a one-click "mark complete" action once the check-in
has happened.

### 11.4 AI-suggested possible findings from a patient photo (MedGemma)

Everywhere else in this codebase, the AI is instructed never to name a condition or diagnosis —
that job stays with the deterministic rules engine and, ultimately, the human reviewer (§11.1).
`suggest_possible_findings()` in `vision.py` is a **deliberate, narrow exception**, added because
the project's brief explicitly asks for image-based disease detection: it uses Google's
**MedGemma-4B-IT**, a medical-domain vision-language model, to produce a short, hedged,
clinician-facing differential from an uploaded photo (e.g. "possibly consistent with contact
dermatitis (redness, no discharge)") — never an assertion, always additional information for the
reviewer to confirm, never a change to the deterministic risk tier.

Three layers keep this exception safe rather than a loophole:

1. **The prompt itself** never asks for assertive phrasing — it explicitly forbids "the diagnosis
   is", "likely/probable/working diagnosis", and "you have", and requires hedged, third-person
   framing ("possibly consistent with...", "may show visible signs associated with...").
2. **The same shared safety filter** every other AI output in this project must pass
   (`llm._safety_filter`) still runs on this output — it hard-blocks prescriptions, dosages, and
   assertive diagnostic claims regardless of what this feature's own prompt asks for. Naming a
   *possible* condition in hedged, disclaimed form is allowed by that filter's design (it only
   blocks the AI *asserting* a diagnosis, not organizing/suggesting information) — see llm.py's
   own docstring on this distinction.
3. **The disclaimer is enforced in code, not just requested in the prompt.** `_ensure_disclaimer()`
   guarantees the fixed line *"AI-suggested possible considerations from the image only - NOT a
   diagnosis. A qualified clinician must examine the patient and confirm before any action is
   taken."* is present, and the reviewer-facing template renders that same fixed disclaimer text
   unconditionally alongside any findings — so the guarantee does not depend on model compliance.

All three layers are covered by pure-Python tests (`tests/test_vision_diagnosis.py`, no model
download needed) that lock this design in.

**Technical honesty note**: MedGemma's image support in `llama-cpp-python` is a genuinely
unsettled area of the ecosystem as of when this was built — the official `abetlen/llama-cpp-python`
package does not yet ship a Gemma-3-aware multimodal chat handler (an open PR, #1989, adds one but
was unmerged at the time of writing). This module tries a third-party fork's `Gemma3ChatHandler`
first (best fidelity: `pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"`),
and automatically falls back to the official package's `Llava15ChatHandler`, which has been
reported to load MedGemma's GGUF + mmproj pair successfully since the underlying vision-projector
format is compatible — so the feature works out of the box once the model files are downloaded,
but (exactly like moondream2's integration below) the fallback handler's exact output quality is
best-effort and worth spot-checking on your hardware before a live demo. Either way, this can only
ever degrade to "feature unavailable" — it never breaks intake, OCR, or the general-purpose image
description.

MedGemma is also gated under Google's "Health AI Developer Foundations" terms of use, unlike every
other model this project downloads anonymously — see the Quick Start section above for the
`HF_TOKEN` flow if the anonymous download is rejected.

**A real crash found on first actual-hardware run, and its fix**: on-device testing on an NVIDIA
Jetson board hit a hard process kill (`Killed`, no Python traceback) the moment CLIP started
encoding an uploaded photo. That signature is the Linux OOM killer, not an application bug - the
process (Qwen2.5 + MedGemma-4B + its BF16 mmproj, all resident at once) ran out of RAM during image
encoding. Three coordinated fixes, all in `triage_engine/vision.py` and `app.py`:

1. **`n_ctx` was 4096, copied from the text model without thinking about it** - a vision call only
   ever produces a couple hundred output tokens, so that was reserving a needlessly large KV-cache.
   Lowered to 2048 (matching moondream2's own setting) and made tunable (`MEDGEMMA_N_CTX`,
   `MEDGEMMA_N_THREADS`, `MEDGEMMA_N_BATCH` env vars) for hardware that needs to go lower still.
2. **`CUDA_VISIBLE_DEVICES=""` is now set at the earliest point in process startup** (`app.py`,
   before any AI module is imported). This project is CPU-only by design, but Jetson boards
   commonly ship CUDA-capable `llama-cpp-python` wheels - and Jetson's *unified* memory
   architecture means a stray GPU allocation draws from the exact same physical RAM the CPU needs,
   so hiding the GPU from the process entirely removes a variable rather than trading memory for
   speed.
3. **Large uploaded photos are now downscaled before being sent to the model at all**
   (`_image_data_uri`, capped by `VISION_MAX_IMAGE_DIM`, default 1024px). Both vision models resize
   to a fixed internal resolution regardless (MedGemma: 896x896, confirmed directly in the crash
   log's own `clip_encode: ... nx=896, ny=896` line) - sending a multi-thousand-pixel phone photo
   bought nothing but extra decode time and peak memory. This is a genuine speed win as well as a
   memory one, with a tested, non-raising fallback to the original file if Pillow can't parse it.

One deliberate tradeoff, made explicitly rather than defaulted into: `vision.preload_async()` (load
the vision model on a background thread at startup, like `llm.py` already does for the text model)
exists but is **off by default** (`VISION_PRELOAD=1` to opt in). Preloading would make the first
photo upload faster, but it also means several extra GB stay permanently resident for the whole
process lifetime, whether or not any given case ever has a photo attached - exactly the kind of
standing memory pressure that caused the crash above. The safer, lazy, load-on-first-photo behavior
this app has always had stays the default; opt into eager preload only on hardware with headroom to
spare. All of this - the tunable constants, the downscaling, and the preload opt-in gate - is
covered by `tests/test_vision_diagnosis.py` without needing the actual weights.

**UPDATE - the three fixes above did NOT fully resolve the crash, and a more fundamental change
was needed (read this before trusting the paragraphs above on their own):** on a second real-
hardware run, with all three fixes live (confirmed via a new log line showing `n_ctx=2048` was
genuinely in effect), the identical crash happened again at the identical point
(`clip_encode: ... nx=896, ny=896` → `Killed`). `dmesg` on the device confirmed a genuine kernel
OOM kill (`Out of memory: Killed process ... anon-rss:4177560kB`, ~4.18GB resident, on a 7.4GB-RAM
board). This makes sense in hindsight: `n_ctx`/batch-size tuning and downscaling the *input* image
both act on the LLM's own context or the pre-encode payload - neither touches CLIP's internal
buffer for its **fixed** 896x896 encode target, which is where the process actually dies. Adding
8-16GB of swap as an OS-level mitigation did stop the hard crash, but that alone traded
crash-safety for a real speed cost (disk-backed paging on Jetson-class hardware is typically
eMMC/SD-backed, not NVMe), which conflicts with this project's own "must be optimized and fast"
requirement even when it no longer crashes outright. Published CPU-only benchmarks for 4B-class
GGUF models on Jetson-class ARM CPUs (roughly 3-4 tokens/sec for decode, plus 10-20+ seconds of
unaccelerated CLIP encode time) put a realistic MedGemma response at **60-90+ seconds per photo**
even without any swap paging - a genuinely poor fit for a live, fast demo on this hardware class,
independent of the crash.

The fix that actually addresses this (see `vision.py`'s module docstring "WHICH ONE LOADS BY
DEFAULT" and `VISION_PREFER_MEDGEMMA` for the full reasoning): **moondream2 (1.9B, known to run on
hardware as constrained as a Raspberry Pi) is now the default active vision model**, not MedGemma.
MedGemma remains fully supported, opt-in, via `VISION_PREFER_MEDGEMMA=1` for a deployment with the
RAM/swap headroom and time budget to want its medically-trained "AI-suggested possible findings"
output - the swap mitigation above is still recommended for that opt-in path. This is a genuine
accuracy-vs-speed tradeoff, not a strictly-better fix: moondream2 has no medical-domain training,
so its output is a plain visual description rather than a hedged clinical differential - but the
hedged-prompt/safety-filter/disclaimer design described just above in this section is prompt
engineering this project built itself, not something unique to MedGemma, so the same guardrails
apply regardless of which model is active. Covered by `tests/test_vision_diagnosis.py`'s
`TestVisionModelPreferenceOrder`.

### 11.5 Experimental local vision model (moondream2) — honestly flagged as unverified

`vision.py` also wires up `moondream2` via `llama-cpp-python`'s `MoondreamChatHandler` to generate a
plain-language, non-diagnostic description of an uploaded photo (e.g. "a close-up photo showing
reddened skin on a forearm with visible swelling" — never a diagnosis). This is now the **default,
preferred vision model** (see §11.4's "UPDATE" note above for why the original "MedGemma preferred"
default was reversed after real-hardware testing): moondream2 loads first unless
`VISION_PREFER_MEDGEMMA=1` is set, in which case MedGemma takes over instead (one model loaded at a
time, to avoid wasting RAM on CPU-only hardware — see `_get_active_vision_model()`).
**This code path could not be run end-to-end during development**, because the build environment
had no internet access to download the moondream2 GGUF files or verify the exact
`llama-cpp-python` chat-handler API surface against a real model file. This already caused one
real, confirmed bug: the original `download_model.py` pointed at `vikhyatk/moondream2`, which
turned out to publish only raw `.safetensors` weights - no GGUF files at all - so every download
attempt 404'd on real hardware. This has been corrected to the actual GGUF conversion repo,
`cjpais/moondream2-llamafile` (`moondream2-050824-q5k.gguf` + `moondream2-mmproj-050824-f16.gguf`,
~2GB combined), verified by directly listing that repo's files rather than guessing filenames
again. The download itself, and the `MoondreamChatHandler` API surface against these exact files,
still have not been exercised end-to-end in the environment that maintains this project - the
fallback behavior (model absent → clear on-screen message, no crash) *is* verified. Before relying
on this feature in a live demo, run `python download_model.py --vision` on a machine with internet
access and confirm it actually loads and produces output.

### 11.6 Translation

`translator.py` provides a curated phrase bank for fixed UI strings, known symptom names, and the
standard follow-up question bank, with **all three languages (English, Hindi, Odia) populated**.
The Odia phrases were written from general language knowledge, using standard everyday
health-communication vocabulary, but were not verified against a live Odia speaker or dictionary
lookup at build time. Free-text patient notes are not machine-translated by this lightweight layer
and are always shown in the original language for reviewer accuracy.

The AI-generated narrative summary is a different problem: it's free-form model output, so it
needs a real machine-translation step, not a phrase bank. This went through two designs:

- **Originally**, the summary was translated by asking Qwen2.5 (the same generalist chat model
  used for extraction/summary - §3) to translate it, with a script-range check and a
  repetition-loop check (`triage_engine/llm.py`, `SCRIPT_RANGES` / `_script_match_ratio` /
  `_looks_degenerate`) run on every output before it's shown. This caught real failures during
  testing - Qwen2.5 occasionally produced fluent-looking **Bengali when asked for Odia** (a related
  but different script) or looped into a repeated phrase - but a caught failure still means the
  summary silently falls back to English. Odia isn't one of Qwen2.5's officially supported
  languages at all, so this was always going to be the weaker case.
- **Now**, `triage_engine/indic_translate.py` adds **IndicTrans2** (AI4Bharat, MIT licensed) - a
  machine-translation model purpose-built and benchmarked specifically for English→Indic
  translation, Odia included - and prefers it for this one task. This is the same principle
  already used for MedGemma vs. a generalist vision model (§11.4): match the model to the job. The
  full chain, per translation, is **IndicTrans2 → Qwen2.5 → English** - each step only runs if the
  previous one is unavailable or fails its quality check (the *same* script-match/repetition checks
  from `llm.py`, reused rather than reimplemented, since the failure modes are model-agnostic), so
  it can only ever get *more* reliable than before, never regress. A failed check at every level
  falls back to showing the English original with an honest on-screen note - never silently shows
  bad output.

**A real, confirmed gap found after this project shipped**: IndicTrans2's own MIT *license* covers
usage rights, but its Hugging Face **repo is separately gated** - it requires logging in, accepting
terms, and authenticating the download with a token, exactly like MedGemma (§11.4), and this was
originally undocumented and unhandled here (the downloader called `snapshot_download()` with no
token). The practical effect: without a token, the download silently 401/403s, IndicTrans2 never
becomes available, and Hindi/Odia translation quietly runs on the Qwen2.5 fallback the whole time -
producing exactly the two symptoms this feature exists to fix (mediocre Hindi, Odia usually falling
back to English) while looking, from the UI, like IndicTrans2 just isn't very good. Fixed by
passing `HF_TOKEN` through to `snapshot_download()` (matching how the single-file downloader
already handles MedGemma's gating) and by making every relevant message - `download_model.py`'s
output, `indic_translate.py`'s docstring, and its `/health` status message - say so explicitly. To
get real IndicTrans2 quality: accept the terms at
https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M, create a token at
https://huggingface.co/settings/tokens, then run
`HF_TOKEN=hf_xxx python download_model.py --indic-translate`.

IndicTrans2 needs heavier, PyTorch-based dependencies (`transformers`, `torch`,
`IndicTransToolkit` - see requirements.txt and the Quick Start section above) that the rest of this
project deliberately avoids requiring by default, since it's the one AI feature here without a
maintained GGUF/llama.cpp build; it is entirely optional and the app runs fine without it, exactly
as it did before this module existed. Like MedGemma and moondream2's integrations, its "model
present" code path was reasoned through carefully but could not be run end-to-end against real
downloaded weights in the environment that built it - confirm it on a machine with internet access
before relying on it in a live demo. This design (including the fallback chain and the
script/degeneracy check reuse) is covered by `tests/test_indic_translate.py` without needing the
actual weights.

### 11.7 Professional dashboard design

The UI is built around a sidebar-shell layout (dark sidebar navigation with role-aware links, a
user card, and section grouping), stat tiles, badges, tables, and forms, and a gradient centered
auth screen for login — all in a single hand-written CSS design system (`static/css/style.css`)
using only system fonts (no CDN dependency, preserving full offline operation and a strict CSP,
§6.4). Verified via automated screenshots and a full Playwright pass (desktop widths) across the
login, dashboard, triage detail (English/Hindi/Odia), audit log, and intake flows.

### 11.8 Other known heuristic limitations

- **Negation handling** (`_is_negated` in `risk_rules.py`) is a fixed-width lookback heuristic, not
  full clinical NLP. It correctly handles common patterns ("no fever", "denies chest pain") but can
  be fooled by more complex sentence structure — which is exactly why raw patient text is always
  shown to the reviewer alongside the structured extraction, never instead of it.
- **Vital-sign thresholds** use adult defaults with a simple age adjustment (infants/elderly get a
  lower bar for escalation); a production system would use proper age-banded pediatric/geriatric
  ranges.

### 11.9 Multilingual voice input

Voice intake (`static/js/intake.js`, browser-native Web Speech API) originally hardcoded
`recognition.lang = "en-IN"` regardless of which language the intake form was actually set to -
meaning a patient describing symptoms in Hindi or Odia would either get garbled English-phonetic
transcription or nothing at all, undermining the whole point of supporting three languages
everywhere else in this app. Fixed: voice input now reads the same "Preferred language" selector
already on the intake form (`triage_engine/translator.py`'s `SUPPORTED_LANGUAGES` - `en`/`hi`/`or`)
and requests the matching BCP-47 tag (`en-IN`/`hi-IN`/`or-IN`) from the browser, re-read on every
recording start so switching languages takes effect immediately without a page reload.

**Honest limitation, not fixed by this change**: which languages the browser's built-in speech
engine actually supports is decided entirely by the browser/OS, not by this app - there is no
official published list, it isn't guaranteed stable, and it can differ between browsers. Hindi is
a major, broadly-supported language and should work reliably in Chrome. Odia is a lower-resource
regional language; support is genuinely uncertain and may vary by browser/device. If the browser
rejects the requested language outright, the UI now shows a clear, specific message ("this
browser's voice input doesn't support Odia - please type instead") rather than failing silently -
typing remains fully supported in all three languages regardless. Confirm on the actual demo
device/browser before relying on Odia voice input specifically in front of judges.

### 11.10 Translation coverage gap on the triage detail page

Switching the triage detail page to Hindi or Odia (the language links at the top of the page)
translated the UI chrome, the fixed follow-up question bank, and - once IndicTrans2/Qwen2.5 was
wired in - the AI-generated summary. It did **not** translate every other AI-generated field on
the same page: chief complaint, missing information, the AI's own extra follow-up questions, and
(when present) the uploaded photo's AI description and AI-suggested findings all stayed in English
regardless of which language was selected - the template even carried an explicit "(English only,
review before use)" label on the extra follow-ups confirming this was a known, undocumented gap
rather than an oversight anyone had caught.

Fixed: `app.py` now runs every one of those fields through the same translation path already used
for the summary (IndicTrans2 first, Qwen2.5 as a generalist fallback, original English as the final
fallback if both are unavailable or fail a quality check), via two small helpers -
`_translate_field()` for a single block of text and `_translate_list_field()` for a list of
strings (missing info, extra follow-ups, findings). Each translation is cached per note per
language in a new `extra_translations_json` column (`triage_engine/database.py`, auto-migrated for
existing databases the same way every other column addition in this project has been - no manual
migration step needed), so re-viewing the same note in the same language re-uses the cached result
instead of re-running a slow CPU-bound model call on every page load. The raw patient-entered text
and OCR'd document text are deliberately left untranslated always, in the original language, for
reviewer accuracy - only AI-*generated* narrative fields are translated.

**Honest limitation**: list fields (missing info, extra follow-ups, findings) are translated by
joining the items with newlines, translating as one block, and splitting the result back on
newlines - a best-effort approach for a small local model, not a guarantee. If the model doesn't
preserve line breaks exactly, item count/order could shift; the existing translation quality
checks (degenerate-repetition and script-match ratio, from `llm.py`) still apply and fall back to
English on a bad result, and the page shows a visible warning when that fallback triggers. This fix
covers the on-screen triage detail page only - the downloadable referral PDF and the plain-text
note summary were out of scope for this request and remain English-only.

### 11.11 Symptom/category phrase-bank gaps and untranslated risk reasons

While verifying the fix above, three more real, un-flagged English-only gaps turned up on the same
page:

- **`SYMPTOM_TRANSLATIONS` covered only 7 of the 27 canonical symptoms** in
  `risk_rules.SYMPTOM_ONTOLOGY`. A patient reporting one of the other 20 - seizure, rash, sore
  throat, injury/fracture, and so on - would see that symptom name fall back to English in the
  "Reported Symptoms" list even with Hindi or Odia selected, because
  `translate_symptom_label()`'s fallback is (deliberately) English when a translation is missing.
  This directly undercuts the point of a patient giving voice input in Hindi or Odia: the system
  understood the symptom, but showed it back in the wrong language. **Fixed**: all 27 symptoms now
  have Hindi and Odia entries (same disclaimer as always - phrase-bank translations, not verified
  against a native speaker; see the ODIA ACCURACY NOTE at the top of `translator.py`), and a new
  `test_translator.py` asserts every symptom `risk_rules.py` knows about has both translations, so
  this can't silently regress if a symptom is ever added to the ontology without one.
- **"Explicitly Denied by Patient" showed raw canonical keys** (`chest_pain`, not even
  humanized to "Chest pain") regardless of language - a bug independent of translation, just never
  caught because this list is empty on most demo notes. **Fixed**: it now goes through the same
  `translate_symptom_label()` lookup as the main symptom list, exactly as the "Rules engine
  detected" cross-check line in the collapsible safety-net panel does.
- **Risk reasons and symptom categories were never translated at all.** The plain-language "why
  this priority" bullets under Risk Priority - the single most prominent AI/rules-generated text
  on the page - and the `category: cardiac`-style label next to each symptom stayed in English no
  matter the language selected. **Fixed**: symptom categories are a small fixed phrase bank
  (`CATEGORY_TRANSLATIONS`, translated instantly with no model call, like symptom names); risk
  reasons are freeform generated sentences, so they go through the same on-demand
  IndicTrans2→Qwen2.5→English cache used for chief complaint/missing info/etc. (§11.10), keyed as
  `"risk_reasons"` in `extra_translations_json`.

### 11.12 Full pipeline audit (pre-hackathon final pass)

Before final delivery, every module in `triage_engine/` and every route in `app.py` was read
end-to-end (not just spot-checked) looking specifically for anything that could still crash,
silently misclassify a case, or leak data across roles. Ten more real, confirmed issues were
found and fixed, each with a new regression test so it can't come back silently:

**Correctness/safety (affects what risk tier a case gets):**
- **OCR'd document text was never scanned for symptoms** - only numeric vitals were read from it
  (`extractor.py`). A scanned referral note that plainly said "unconscious, severe bleeding"
  produced NO red-flag detection if the typed-symptom box was left empty (a real scenario: an
  image-only upload). Now both sources are scanned together.
- **A short, generic AI-reported symptom name could false-match a much more specific red-flag
  alias** (`pipeline.py`'s `_match_ontology()`) - "weakness"/"weak" matched `stroke_signs` via its
  alias "sudden weakness one side", "bleeding" matched `severe_bleeding`, "speech" matched
  `stroke_signs` again - silently forcing a false "emergency" tier for routine complaints.
  Matching is now one-directional (the alias must appear inside the AI's name, never the
  reverse), which still resolves genuine paraphrases correctly.
- **The blood-pressure regex matched ANY "N/N" fragment** with no plausibility check - a pain
  score ("12/10 unbearable") or a date ("follow up on 15/11") was silently read as a blood
  pressure reading and flagged "abnormal". Now bounded to physiologically plausible values
  (systolic > diastolic, both within a real human range).
- **Age-based sensitivity only recognized "months" and "years"** - ANY age in months was treated
  as "<1 year" regardless of the number ("18 months" mislabeled "infant"), and "days" wasn't
  handled at all, so a real newborn described in days ("75 days") fell through to being read as
  75 *years* and mislabeled "elderly". Ages are now normalized to fractional years first.
- **Compound "does not have X or Y" negation** only excluded X, the item directly next to the
  negation cue - Y was wrongly reported as present. The negation check now widens specifically for
  a symptom directly chained by "or"/"and"/"nor" (never across a comma, "but", or "however", to
  avoid the opposite, more dangerous mistake of wrongly negating a genuinely-present symptom).
- **Negated symptoms couldn't be translated** in either extraction path: the rules path
  pre-humanized them (`"chest_pain"` → `"chest pain"`) before storage, which no longer matched
  `SYMPTOM_TRANSLATIONS`'s underscored keys; the AI path never mapped its free-text negated-symptom
  strings back to the ontology at all (present symptoms were already mapped this way). Both now
  stay canonical until display time, exactly like present symptoms.

**Crash / reliability:**
- **The referral PDF generator crashed on ordinary text.** Reviewer notes, AI summaries, and other
  free text were inserted directly into reportlab's `Paragraph()`, which parses its input as a
  small XML dialect - an unescaped `<` not immediately followed by a matching close tag (e.g. a
  reviewer typing "improving<discharge planned") raised an uncaught exception with no test
  coverage at all. All free text is now escaped before use.
- Added a proper styled error page for **413 (file too large)**, matching every other error code.

**Security / access control:**
- **Open redirect on `/login`'s `next=` parameter** - a crafted link with an off-site `next` would
  send a victim to an attacker-controlled site immediately after a legitimate sign-in. Now
  restricted to same-site relative paths only.
- **Any intake worker could view any OTHER patient's full reviewer detail page** (chief complaint,
  AI summary, risk reasons, uploaded-photo AI findings) across every facility, just by editing the
  note ID in the URL - the route granted the `intake_worker` role access with no check tying the
  note to the worker who created it. A new `created_by_user_id` column (auto-migrated, like every
  other column added in this project) now scopes that access to a worker's own submissions;
  reviewers are unaffected and can still open any note.

**Usability:**
- **The intake form lost everything on a validation error** (e.g. an invalid facility type) -
  every field, including a long typed symptom description, had to be retyped from scratch. Form
  values are now preserved and restored.
- Regenerating the AI summary reset the summary's translation cache but not the cache for chief
  complaint/missing info/follow-ups/risk reasons, so a reviewer could see a stale Hindi/Odia
  translation next to freshly regenerated English text with no warning. Both caches now clear
  together.
- Fixed a stale UI hint claiming voice input was English-only (it has followed the selected
  language since §11.9), and removed a leftover unused database query on every page view.

**Reviewed and found sound, no changes needed:** `triage_engine/security.py` (CSRF, rate
limiting, password migration), `triage_engine/audit.py`, `config.py`, `triage_engine/vision.py`,
`triage_engine/indic_translate.py`, `triage_engine/database.py`'s migration mechanism, and
`triage_engine/llm.py`'s safety filter and grammar-constrained JSON extraction.

**Known, deliberately-accepted limitations** (documented rather than "fixed" - the trade-off was
judged not worth the added complexity for a hackathon prototype): the referral PDF and the intake
confirmation page do not apply Hindi/Odia translation, unlike the triage detail page (§11.10);
concurrent requests translating the same note in different languages at the same moment could
each overwrite the other's cache write (no data corruption or crash, just a recomputed
translation next time); comma is deliberately not treated as a negation-list connector because
it's genuinely ambiguous between "another denied item" and "a new clause" (§11.12's negation fix).

### 11.13 The triage detail page's language switcher only translated 4 headings

A real, confirmed bug found after a specific report ("Odia output isn't working properly" on the
reviewer detail page): `translator.UI_STRINGS` and the `{{ ui.* }}` lookups in
`templates/triage_detail.html` only ever covered 4 headings (chief complaint, risk priority,
missing information, follow-up questions). Every OTHER section heading, button label, placeholder,
and static sentence on that page - "Patient (anonymised view)", "AI-Generated Summary", "Reported
Symptoms", "Timeline (heuristic)", "Vitals", "Reviewer Actions", "Save Review", "Download Referral
PDF", "Record", the field labels under it, and more - was hardcoded English directly in the
template, so switching the language selector to Hindi or Odia left most of the page in English
regardless of the selected language. Two further, more visible problems were part of the same
gap: the **risk-tier badge at the very top of the page** ("Emergency - See Immediately", etc.) and
the **note status** ("New", "In review", ...) were read straight from `RISK_TIER_LABELS`/
`STATUS_FLOW`, which only ever hold English text, with no translation step at all - meaning the
single most prominent piece of text on the page never changed language.

Fixed by adding ~45 new `UI_STRINGS` entries (one per static label/button/sentence), plus two new
lookup dicts and helper functions - `RISK_TIER_TRANSLATIONS`/`translate_tier_label()` and
`STATUS_TRANSLATIONS`/`translate_status_label()` - and rewiring every hardcoded string in
`triage_detail.html` to go through one of these. Two smaller, adjacent gaps in the same phrase-bank
family were fixed alongside it: `FOLLOWUP_TRANSLATIONS_HI`/`_OR` originally covered only 6 of the
~29 distinct follow-up-question strings `extractor.py`'s `CATEGORY_FOLLOWUPS` can actually produce
(cardiac and infection were covered; respiratory, neuro, gi, obstetric, pediatric, trauma,
mental_health, occupational, general, derm, ent, urinary, and musculoskeletal were not) - all are
now covered; and the Timeline section's fixed "duration not specified" literal (shown whenever no
duration could be parsed from the raw text) is now translated too. Covered by
`tests/test_translator.py::TestFollowUpQuestionTranslationCompleteness`,
`TestTierAndStatusTranslation`, `TestUiStringCompleteness`, and
`tests/test_app_integration.py::TestStaticUiStringsTranslatedOnTriageDetail` (13 new tests, 168 ->
181 total).

---

## 12. Privacy & responsible-AI notes

- Use **synthetic or public sample data only**. Nothing in this repository is a real patient record.
- Patient name is optional at intake; everywhere in the reviewer UI it is shown masked
  (`audit.mask_name`), and all cross-references use the generated Triage ID.
- Every view, edit, login, and PDF export is written to `audit_log` (see the Audit Log page).
- The auditor role has a one-click **"Purge Closed Records"** control as a stand-in for a real
  data-retention policy.
- Passwords are hashed, the Flask secret key is generated and stored locally (never hardcoded),
  and neither should be committed to source control — see `.gitignore` (§6).
- **AI-specific controls**: patient-entered free text is treated as untrusted input when it's
  placed into an AI prompt (fenced off with explicit "ignore any instructions in here" framing, to
  reduce prompt-injection risk); AI output (extraction, summary, image description, and
  translation) is filtered for diagnosis/prescription/dosage-shaped language before display; every
  AI-generated field is labeled with which model produced it and when; the risk tier is always
  computed by the deterministic engine regardless of what the AI reports (§11.1-§11.2); and
  generating, regenerating, or viewing any AI output is written to the audit log like any other
  action.
- **The one narrow exception**, MedGemma's AI-suggested possible findings from a photo (§11.4), is
  still never a diagnosis: it is hedged, always shown with a fixed, code-enforced disclaimer,
  never feeds into the deterministic risk tier, and is always additional information the human
  reviewer must confirm by direct examination — the same "AI organizes/suggests, a person decides"
  principle as everywhere else in this project, applied to one more input.
