# Installation & Setup Guide — Healthcare Triage Assistant

This is the complete, step-by-step guide to installing, running, and demoing this project
from a fresh machine — every AI model, every optional feature, every environment variable,
and fixes for the real issues found while building and testing it (including on
constrained, Jetson-class, ~7.4GB RAM CPU-only hardware, and separately during a full
pipeline audit before hackathon delivery).

**The app works fully offline after setup, and works with zero AI models installed at
all** — it falls back automatically to a deterministic, rules-based pipeline. Sections 0–3
below are all you need to get a working demo running in a few minutes; everything from
Section 4 onward is optional, additive, and safe to skip or come back to later.

---

## 0. Prerequisites

- **Python 3.9 or newer.** Check with `python3 --version` (Windows: `python --version`).
- **pip** (comes with Python) and, ideally, **git** to clone/extract the project.
- **~2 GB free disk** for the base install; add up to **~8 GB more** if you enable every
  optional AI model in Section 4 (a breakdown is given there so you only download what
  you actually need).
- OS-level packages required for OCR (reading text from uploaded lab reports/prescriptions):

  **Debian/Ubuntu (including Jetson's Ubuntu base):**
  ```bash
  sudo apt update && sudo apt install -y tesseract-ocr poppler-utils
  ```
  **macOS (Homebrew):**
  ```bash
  brew install tesseract poppler
  ```
  **Windows:**
  1. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki (the UB-Mannheim
     installer is the standard choice) and note the install path (usually
     `C:\Program Files\Tesseract-OCR`).
  2. Install poppler for Windows from https://github.com/oschwartz10612/poppler-windows/releases
     and add its `bin` folder to your `PATH` (needed for PDF report uploads).
  3. If `pytesseract` can't find Tesseract automatically, set the path explicitly in `.env`
     (see Section 5) or at the top of `triage_engine/ocr.py`.

  The app still runs without these — OCR-dependent features (extracting text from an
  uploaded lab report photo/PDF) will show a clear "OCR unavailable" message instead of
  crashing, but you'll want them installed for a full demo.

---

## 1. Get the project and install Python dependencies

```bash
cd triage-assistant          # or wherever you extracted the project
python3 -m venv .venv        # optional but recommended: keeps dependencies isolated
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs Flask, Pillow, OCR bindings, ReportLab (for referral PDF export),
`llama-cpp-python` (the CPU inference engine used by every AI model in Section 4), and the
production-hardening packages (`python-dotenv`, `gunicorn`). It does **not** install the
heavier `transformers`/`torch` stack — that's only needed for dedicated Hindi/Odia
translation (Section 4c) and is commented out in `requirements.txt` by default to keep the
base install lightweight and fast.

If `pip install` fails specifically on `llama-cpp-python`: prebuilt CPU wheels exist for
Windows/macOS/Linux on both x86_64 and arm64, so this normally does **not** require a C++
compiler — a failure here is almost always a Python version too old/new for the available
wheel, or no internet access at install time. The rest of the app still works without it
(rules-based pipeline only); you can retry this one package later.

---

## 2. First run

```bash
./run.sh              # macOS/Linux
```
or, on Windows (or if you want to see each step explicitly):
```bash
python seed.py         # one-time only: creates the database, demo users, and synthetic demo cases
python app.py           # every time you start the server
```

Then open **http://localhost:5000** in a browser. At this point the app is fully
functional using the deterministic rules-based pipeline — no AI model download is
required to use every core feature (intake, risk triage, reviewer dashboard, referral PDF
export, multilingual UI).

`run.sh`/`seed.py` only seed the database if `data/triage.db` doesn't already exist, so
running them again later is always safe (it won't wipe your data).

---

## 3. Demo accounts

All passwords are `demo123`.

| Username    | Role                       | What they can do                                     |
|-------------|----------------------------|-------------------------------------------------------|
| `worker1`   | Health Worker (intake)     | Record consent + patient-reported symptoms/reports    |
| `reviewer1` | Medical Officer (reviewer) | Dashboard, triage detail, sign-off, referral export    |
| `auditor1`  | Facility Data Officer      | Audit log, data-minimisation ("purge closed")          |

### A 2-minute demo script (good for a hackathon walkthrough)

1. Sign in as `worker1` → accept consent → New Intake.
2. Type (or use the 🎤 voice button — try switching the language dropdown to Hindi or
   Odia first) something like: *"Patient has severe chest pain and breathlessness since
   this morning."* → Submit. Notice the case is immediately triaged as **Emergency** with
   plain-language reasons shown.
3. Optionally attach a photo (a wound, rash, or skin condition) or a PDF report — this
   demonstrates OCR + the vision model's photo description together.
4. Log out, sign in as `reviewer1` → open the **Dashboard** → the new case appears at the
   top (emergency cases always sort first) → click **Review**.
5. On the triage detail page: point out the risk tier + reasons, the rules-engine
   cross-check panel (click to expand — shows the deterministic safety net independently
   agrees), and switch the language links at the top to Hindi/Odia to show the whole page
   (including AI-generated fields) re-render in that language.
6. Fill in reviewer notes, mark the case **Reviewed**, and download the **Referral PDF** —
   a clean, printable handoff document.
7. Log out, sign in as `auditor1` → **Audit Log** — every view/edit of every case is
   logged with a timestamp, demonstrating the accountability trail.

---

## 4. Enabling AI features (all optional, one-time downloads)

Nothing in this section is required for the demo script above to work — it already runs
on the rules-based pipeline. Enable these to show the "AI-powered" badge and richer,
model-generated narrative text instead.

### Which model do I actually need?

| I want to demo... | Download this | Approx. size |
|---|---|---|
| AI-written chief complaint/summary from patient text | 4a (Qwen2.5) | ~1 GB |
| A plain-language description of an uploaded photo | 4b, moondream2 (default) | ~2 GB |
| An AI-suggested (hedged) list of possible visible findings from a photo | 4b, MedGemma (opt-in) | ~4.2 GB |
| Higher-quality Hindi/Odia translation | 4c (IndicTrans2) | ~1 GB + ~2-3 GB for `torch` |

### 4a. Text extraction & summarization (Qwen2.5) — recommended first

```bash
python download_model.py     # ~1 GB, needs internet, anonymous (no login/token needed)
```

Once downloaded, restart the app — AI-powered symptom extraction and summarization
activate automatically. This is the model used for every text-based case, and is also
used as the translation fallback if IndicTrans2 (4c) isn't installed.

### 4b. Photo/image understanding (vision models)

Two vision models are supported; **only one loads at a time** to conserve memory on
CPU-only hardware.

**moondream2 — the default, fast, general-purpose model** (plain-language photo
description, no medical training):

```bash
python download_model.py --vision-only     # ~2 GB, anonymous, no login needed
```

This is what loads automatically once downloaded — no extra configuration needed.

**MedGemma-4B — opt-in, medical-domain model** (produces an AI-suggested, hedged
differential of possible visible conditions, not just a description):

```bash
python download_model.py --medgemma        # ~4.2 GB, GATED — see below
```

MedGemma's repo requires Hugging Face authentication even though the model itself is
free to use:
1. Visit https://huggingface.co/unsloth/medgemma-4b-it-GGUF, log in, accept the terms shown.
2. Create a token at https://huggingface.co/settings/tokens ("Read" access is enough).
3. Re-run with the token:
   ```bash
   HF_TOKEN=hf_your_real_token_here python download_model.py --medgemma
   ```

**Important:** downloading MedGemma is not enough to activate it — by default this
project prefers the faster, lighter moondream2 for safety on constrained hardware. To
make MedGemma the active vision model (needed for the AI-suggested-findings feature),
set this before starting the app:

```bash
VISION_PREFER_MEDGEMMA=1 ./run.sh
```

**Know the tradeoff before enabling this on constrained hardware:** MedGemma is roughly
4x heavier than moondream2 and its image-encoding step has caused real out-of-memory
crashes on 7–8GB RAM boards. If you enable it, also add swap space as a safety margin:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Even with swap preventing a crash, expect MedGemma to take roughly **60–90+ seconds per
photo** on Jetson-class CPUs — moondream2 typically responds in single-digit seconds. On a
typical laptop CPU (recent Intel/AMD/Apple Silicon), expect both models to be noticeably
faster than the Jetson figures above.

### 4c. Dedicated Hindi/Odia translation (IndicTrans2) — optional quality upgrade

Without this, Hindi/Odia translation of AI-generated text uses the generalist Qwen2.5
model, which has mediocre Hindi and often fails to produce usable Odia at all (falls back
to English with a visible warning). IndicTrans2 fixes both.

```bash
pip install transformers torch IndicTransToolkit
```

This model's repo is **also gated** (despite being MIT licensed):
1. Visit https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M, log in, and
   click "Agree and access repository."
2. Create a token at https://huggingface.co/settings/tokens (or reuse the one from 4b).
3. Download with the token:
   ```bash
   HF_TOKEN=hf_your_real_token_here python download_model.py --indic-translate
   ```

No further configuration needed — once downloaded, it's automatically preferred over
Qwen2.5 for Hindi/Odia translation of every AI-generated field (summary, chief complaint,
missing info, extra follow-ups, photo description/findings, and risk reasons).

**Getting your Hugging Face token, in detail (if you're unfamiliar):**
- Go to https://huggingface.co/settings/tokens → "Create new token" → any name, "Read"
  role → Create.
- The full token is shown **once**, with a copy icon next to it — copy it immediately.
  If you navigate away before copying, you can't view the full value again; you'd need
  to click the token's menu → "Invalidate and refresh" to get a new value shown once
  more (this keeps the same token entry, no need to delete and recreate it).
- In the terminal, paste with `Ctrl+Shift+V` (plain `Ctrl+V` often doesn't work in Linux
  terminals) right after `HF_TOKEN=`.
- Never commit a real token to git or share it in a screenshot — treat it like a password.

---

## 5. Making settings persistent with `.env`

Rather than prefixing environment variables on every run, copy `.env.example` to `.env`
and set values there — `run.sh`/`app.py` load it automatically:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (none) | Auth token for gated downloads (MedGemma, IndicTrans2) |
| `VISION_PREFER_MEDGEMMA` | `0` | `1` = use MedGemma as the active vision model instead of moondream2 |
| `VISION_PRELOAD` | `0` | `1` = load the vision model at startup instead of on first photo (uses more RAM permanently) |
| `LLM_N_THREADS` | CPU count − 1 | Threads for text-model inference; lower on shared/constrained machines |
| `MEDGEMMA_N_THREADS` | CPU count − 1 | Same, specifically for MedGemma |
| `VISION_MAX_IMAGE_DIM` | `1024` | Downscale uploaded photos to this max dimension before sending to the vision model |
| `MAX_UPLOAD_MB` | `12` | Max size for an uploaded report photo/PDF before the app rejects it with a friendly error |
| `SECRET_KEY` | auto-generated, persisted to `data/.secret_key` | Flask session signing key — auto-created on first run; don't need to set this yourself |
| `APP_ENV` | `development` | Set to `production` when deploying for real (stricter cookie/session settings) |

See `.env.example` itself for the full list with explanations.

---

## 6. Verifying everything works

Visit **http://localhost:5000/health** while the app is running. It returns a live JSON
status of the database and every AI feature (text extraction, vision description,
AI-suggested findings, translation) — each with a plain-English message explaining
exactly what's active and what isn't, rather than you having to guess from the UI. This
is the fastest way to confirm a model download actually worked before a live demo.

---

## 7. Running the test suite (optional, for verifying code changes)

```bash
python -m unittest discover -s tests -v      # zero extra dependencies, ~10 seconds
# or
pytest
```

All 168 tests should pass. None of them need any model downloaded — AI/vision/translation
behavior is mocked where relevant, so the suite runs the same way whether or not you've
done Section 4. If you make any code changes before a demo, re-run this first.

---

## 8. Docker (optional, alternative to the steps above)

```bash
docker compose up --build -d
docker compose exec triage-assistant python download_model.py   # then any of the flags from Section 4
```

`docker-compose.yml` mounts `./data`, `./models`, and `./logs` as volumes so the
database, downloaded weights, and logs all persist across container rebuilds. The
container runs as a non-root user and seeds demo data automatically on first boot only
(never overwriting an existing database).

---

## 9. Resetting / starting over

To wipe all data and start fresh (new demo accounts, no cases):
```bash
rm data/triage.db
python seed.py
```
This never touches downloaded model files in `models/` — those are safe to keep.

To remove everything, including downloaded models (to free disk space):
```bash
rm -rf data/triage.db models/ logs/*.log
```

---

## 10. Troubleshooting

**Download fails with a 401/403 error.** The model's repo is gated — you need an
`HF_TOKEN` (see Sections 4b/4c). A plain 404 instead means the repo/filename is wrong;
that shouldn't happen with the models documented above, but if it does, the exact
Hugging Face repo and filenames are listed in the comments at the top of
`download_model.py`.

**Photo upload crashes the app / process gets `Killed`.** This is the Linux kernel's
out-of-memory killer, not an application bug — confirm with
`sudo dmesg | tail -30 | grep -i "killed process\|out of memory"`. Fixes, in order:
1. Make sure `VISION_PREFER_MEDGEMMA` is **not** set to `1` unless you've added swap
   (moondream2, the default, is much lighter and shouldn't trigger this).
2. Add swap space (see the commands in Section 4b) as a safety margin.
3. Lower `MEDGEMMA_N_CTX`/`MEDGEMMA_N_BATCH` further via `.env` if still using MedGemma.

**AI features seem unavailable even though I downloaded the model.** Check
`/health` (Section 6) first — it tells you exactly why (missing files, failed to load,
wrong preference setting) instead of you having to search logs.

**Translations to Hindi/Odia look poor, or fall back to English.** This is expected
behavior, not a bug, when IndicTrans2 isn't installed (Qwen2.5 is a much weaker
translator, especially for Odia) — a visible warning banner explains this on the page.
Check `/health`'s `ai_indic_translation_model` field to confirm whether IndicTrans2 is
actually active.

**"OCR unavailable" on every upload.** `tesseract-ocr` (and, for PDFs, `poppler-utils`)
aren't installed at the OS level — see Section 0. This is a system package, not a Python
package, so `pip install` alone won't fix it.

**A file upload is rejected as "too large."** Increase `MAX_UPLOAD_MB` in `.env`
(Section 5), then restart the app.

**Port 5000 is already in use.** Another process (on macOS, often AirPlay Receiver) is
using it. Either stop that process, or run on a different port:
`python app.py` reads the `PORT` env var — e.g. `PORT=5050 python app.py`.

**Tests fail after pulling new code.** Run `pip install -r requirements.txt` again in
case a new dependency was added, then re-run Section 7. If a specific test fails, its
name and docstring explain exactly what real bug it's guarding against — read that first.
