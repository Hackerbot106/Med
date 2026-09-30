"""
Local, offline, CPU-only VISION models for image understanding of uploaded
photos (e.g. a visible rash, wound, swelling, or skin lesion photographed at
intake) - the "image understanding" part of the multimodal capability the
problem statement asks for, and the home of the newer "AI-suggested possible
findings from a patient photo" feature requested for this project.

TWO MODELS, TWO DIFFERENT JOBS (read this before changing anything here):

  1. moondream2 - general-purpose small vision-language model. Used ONLY for
     `describe_image()`: a purely OBJECTIVE description of what is visible
     (colour, swelling, size, discharge) for the reviewer's own assessment.
     It is explicitly instructed never to name a condition or diagnosis.

  2. MedGemma-4B-IT - Google's medical-domain vision-language model
     (https://huggingface.co/unsloth/medgemma-4b-it-GGUF), used for the new
     `suggest_possible_findings()`: an AI-SUGGESTED, hedged differential of
     possible visible conditions, explicitly framed as "for clinician
     confirmation, not a diagnosis" and passed through the SAME safety
     filter (`llm._safety_filter`) as every other AI output in this project.
     This is a deliberate, narrow exception to this project's "AI never
     names a condition" rule elsewhere - see the SAFETY DESIGN section below
     for exactly how it is still kept non-final and human-reviewed.

Both models are loaded lazily, are entirely optional, and any failure to
load or run either one is caught and degrades gracefully - a missing/failed
vision model never breaks the rest of the app; the photo is simply attached
for the reviewer to look at directly, exactly as if this file didn't exist.

Only ONE vision model is loaded into memory at a time - this matters on
CPU-only / memory-constrained hardware, where loading two ~2-4GB models
simultaneously would be wasteful. See `_get_active_vision_model()`.

WHICH ONE LOADS BY DEFAULT (revised after real hardware testing - read this
before assuming MedGemma is active): moondream2 is preferred by default
(`VISION_PREFER_MEDGEMMA=0`, the default). This was changed FROM "MedGemma
preferred" after testing on a real Jetson-class 7.4GB-RAM board surfaced two
concrete problems with defaulting to MedGemma: (1) its image-encoding step
(`clip_encode`, a fixed 896x896 resize regardless of input size) triggered a
genuine kernel OOM-kill under memory pressure - see the git history /
README §11.4 "A real crash found on first actual-hardware run" - and (2)
even once that crash is worked around (e.g. with added swap), published
CPU-inference benchmarks for 4B-class GGUF models on Jetson-class ARM CPUs
put MedGemma in the range of roughly 60-90+ seconds per photo, which is a
poor fit for this project's own "must be optimized and fast" requirement.
moondream2 (1.9B, a much smaller fixed encode resolution) is known to run
on hardware as constrained as a Raspberry Pi and returns results in
single-digit seconds - it has no medical-domain training, but this
project's hedged-prompt + safety-filter + code-enforced-disclaimer design
(see SAFETY DESIGN below) is prompt engineering built by this project, not
something unique to MedGemma, so it still produces appropriately hedged
output regardless of which model is underneath.

MedGemma remains fully supported as an OPT-IN "deeper analysis" mode -
set `VISION_PREFER_MEDGEMMA=1` (and restart the app) on a deployment with
enough RAM/swap headroom, or for a deliberate demo step where the extra
latency is expected and communicated up front. This is also the only way
to enable `suggest_possible_findings()` (the AI-suggested-findings
feature), since that feature is MedGemma-only by design - moondream2 has
no medical training to draw on for it. See `VISION_PREFER_MEDGEMMA` below.

SAFETY DESIGN for `suggest_possible_findings()` (read before changing the
prompt or the parsing logic):
  - The model is asked for a SHORT, HEDGED list of possible visible
    considerations ("may be consistent with", never "you have" or "the
    diagnosis is"), each on its own line, always ending with a fixed
    disclaimer. The disclaimer is not merely requested in the prompt: this
    module APPENDS it in code (`_ensure_disclaimer()`) if the model ever
    omits it, so the guarantee does not depend on the model's compliance.
  - Output still passes through `llm._safety_filter()`, which hard-blocks
    prescriptions, dosages, second-person "you have X" language, and
    assertive diagnostic claims ("the diagnosis is", "likely diagnosis",
    etc.) regardless of what this module's own prompt asks for. Naming a
    POSSIBLE condition in hedged, disclaimed form (e.g. "possibly consistent
    with contact dermatitis") is not blocked by that filter, by design - see
    llm.py's own docstring on `_diagnosis_mentions_unattributed`. This is the
    intended, narrow exception: an AI SUGGESTION for the reviewer to
    confirm, never an assertion.
  - This output is stored and displayed as clearly separate from, and
    additional to, the deterministic risk tier - it never feeds back into
    risk_rules.py and can never change the risk tier or bypass reviewer
    sign-off. It is one more input for the human reviewer, not a
    replacement for one.
  - If the model is unavailable, `suggest_possible_findings()` returns
    (None, None) - same as every other best-effort AI feature in this app.

HONESTY NOTE: MedGemma's multimodal (image) support in llama-cpp-python is a
genuinely unsettled area as of when this was written. The upstream,
officially-published `abetlen/llama-cpp-python` package does not (yet) ship
a Gemma-3-aware multimodal chat handler (see llama-cpp-python PR #1989,
open/unmerged). Two ways to get MedGemma images working are supported here,
in order of preference:
  (a) `Gemma3ChatHandler` - correct chat template for Gemma-3-family models,
      only available via a third-party fork, e.g.:
        pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
      Install this FIRST if you want the best-fidelity results.
  (b) `Llava15ChatHandler` - ships in the OFFICIAL llama-cpp-python package
      (>=0.2.90, already a requirement of this project) and has been
      reported to load MedGemma's GGUF + mmproj pair successfully, since
      the underlying vision projector format is compatible. This is used
      automatically as a fallback if (a) isn't installed, so MedGemma image
      analysis works out of the box once the model files are downloaded -
      but the exact prompt/image-token markup LLaVA expects is not
      identical to Gemma-3's native template, so output quality with this
      fallback handler is best-effort and should be spot-checked, exactly
      like moondream2's integration below already was for this project.
Either way, this degrades to "feature unavailable" (never a crash) if
neither handler can be imported or the model fails to load.

Model download: `python download_model.py --medgemma` (see that file). Note
MedGemma sits under Google's "Health AI Developer Foundations" terms of
use - unlike the other models this project downloads anonymously, you may
need to accept those terms on Hugging Face and pass a token (HF_TOKEN env
var) for the download to succeed. The app works fine without it; this
feature simply stays unavailable until the files are present.
"""

import logging
import os
import re
import threading

# Force CUDA invisible to this process BEFORE llama_cpp is ever imported (see _try_load_medgemma) -
# app.py already does this earlier at process startup for the whole app; this is a safety net for
# whenever this module is imported/used on its own (a script, a test) without going through app.py.
# See app.py's own comment on this line for why: this project is CPU-only by design, and a vision
# projector's image-encoding step was observed to be involved in a real out-of-memory process kill
# on hardware (e.g. NVIDIA Jetson boards) that ships a CUDA-capable llama-cpp-python build.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from . import llm  # reuse _safety_filter, _env_int, and the same models/ directory convention

logger = logging.getLogger("triage_engine.vision")

MODELS_DIR = llm.MODELS_DIR

# MedGemma-specific CPU inference tuning, env-overridable like llm.py's own N_CTX/N_THREADS/N_BATCH
# (see that module's "CPU inference tuning" section) - a real out-of-memory kill was observed on a
# Jetson-class board during this project's own testing, right at the image-encoding step, with the
# original n_ctx=4096 (copied from the text model without thinking about it). A vision-findings/
# description call only ever produces a couple hundred output tokens, so a large context bought
# nothing here except a bigger KV-cache - lowered to 2048 (matching moondream2's own setting) and
# made tunable so a memory-constrained deployment can go lower still without touching code.
MEDGEMMA_N_CTX = llm._env_int("MEDGEMMA_N_CTX", 2048)
MEDGEMMA_N_THREADS = llm._env_int("MEDGEMMA_N_THREADS", max(1, (os.cpu_count() or 2) - 1))
MEDGEMMA_N_BATCH = llm._env_int("MEDGEMMA_N_BATCH", 512)

# Preloading the vision model at startup (like llm.py does for the text model) would make the
# FIRST photo upload faster, at the cost of keeping several extra GB permanently resident for the
# app's whole lifetime, whether or not any given case even has a photo. Given a real OOM kill was
# observed on constrained hardware (see vision.py's module docstring), this defaults to OFF - the
# safer, lazy-load-on-first-photo behavior this app has always had - and is opt-in for deployments
# with enough headroom to want the faster first upload.
VISION_PRELOAD_ENABLED = os.environ.get("VISION_PRELOAD", "0") == "1"

# Which model _get_active_vision_model() tries FIRST when both are present - see the module
# docstring's "WHICH ONE LOADS BY DEFAULT" section for the full reasoning. Defaults to moondream2
# (faster, safer on constrained hardware); set VISION_PREFER_MEDGEMMA=1 to opt back into MedGemma
# as the primary model (needed for the AI-suggested-findings feature - see
# findings_availability_status() below).
VISION_PREFER_MEDGEMMA = os.environ.get("VISION_PREFER_MEDGEMMA", "0") == "1"

# --- moondream2: general-purpose, purely descriptive, always-safe fallback ---
VISION_MODEL_FILENAME = "moondream2-050824-q5k.gguf"
VISION_PROJECTOR_FILENAME = "moondream2-mmproj-050824-f16.gguf"
VISION_MODEL_PATH = os.path.join(MODELS_DIR, VISION_MODEL_FILENAME)
VISION_PROJECTOR_PATH = os.path.join(MODELS_DIR, VISION_PROJECTOR_FILENAME)
VISION_MODEL_LABEL = "moondream2 (local CPU vision model)"

# --- MedGemma: medical-domain, used for the AI-suggested findings feature ---
MEDGEMMA_MODEL_FILENAME = "medgemma-4b-it-Q4_K_M.gguf"
MEDGEMMA_PROJECTOR_FILENAME = "mmproj-BF16.gguf"
MEDGEMMA_MODEL_PATH = os.path.join(MODELS_DIR, MEDGEMMA_MODEL_FILENAME)
MEDGEMMA_PROJECTOR_PATH = os.path.join(MODELS_DIR, MEDGEMMA_PROJECTOR_FILENAME)

_active_llm = None          # the one loaded vision Llama() instance, whichever model it is
_active_kind = None         # "medgemma" | "moondream" | None
_active_label = None        # human-readable label, reflects which chat handler MedGemma used
_load_attempted = False
_load_error = None

MEDGEMMA_DISCLAIMER = (
    "AI-suggested possible considerations from the image only - NOT a diagnosis. "
    "A qualified clinician must examine the patient and confirm before any action is taken."
)

DESCRIBE_PROMPT = (
    "Describe ONLY the objective, visible characteristics of this photo relevant to a health "
    "worker's visual triage assessment: approximate location on the body if visible, color, "
    "swelling, size, discharge, or obvious injury. Keep it to 2-3 short sentences. "
    "Do NOT name any medical condition, disease, or diagnosis. Do NOT suggest any treatment. "
    "If the image is not a photo of a body part or visible symptom (e.g. it's a lab report or "
    "document), say so plainly instead of describing it as a symptom."
)

# Deliberately avoids every HARD_DISALLOWED_PATTERNS phrase in llm.py (e.g. "the diagnosis is",
# "likely/probable/working diagnosis", "you have") - it asks for a hedged, clearly-AI-attributed,
# clinician-facing differential instead, which that filter allows through by design.
FINDINGS_PROMPT = (
    "You are assisting a health worker, NOT the patient directly. Look at this photo of a visible "
    "symptom (skin, wound, eye, swelling, etc.) and list up to 4 possible visible considerations a "
    "clinician might want to examine for, based ONLY on what is visibly shown.\n"
    "Format: one consideration per line, each starting with '- ', phrased as a hedge such as "
    "'possibly consistent with ...' or 'may show visible signs associated with ...', with a short "
    "(under 12 words) note of the visual feature that suggests it.\n"
    "Rules:\n"
    "- These are AI-generated suggestions for a clinician to confirm, never a confirmed diagnosis; "
    "never write the phrases 'the diagnosis is', 'likely diagnosis', 'probable diagnosis', "
    "'working diagnosis', or address the viewer as 'you have'.\n"
    "- Do NOT suggest, recommend, or mention any treatment, medication, or dosage.\n"
    "- If the image is not a usable photo of a visible symptom (e.g. a document, lab report, or "
    "an image with nothing clinically visible), reply with exactly: 'Not applicable - no visible "
    "symptom in this image.' and nothing else.\n"
    "- Output ONLY the list (or the not-applicable line), nothing before or after it."
)

MAX_FINDINGS = 4
MAX_FINDING_LEN = 160


def model_files_present():
    """Back-compat name: True if the moondream2 (general-purpose) files are present."""
    return os.path.exists(VISION_MODEL_PATH) and os.path.exists(VISION_PROJECTOR_PATH)


def medgemma_files_present():
    return os.path.exists(MEDGEMMA_MODEL_PATH) and os.path.exists(MEDGEMMA_PROJECTOR_PATH)


def _try_load_medgemma():
    """Attempt to load MedGemma with the best available chat handler. Returns
    (Llama instance, label) on success, or (None, error_message) on failure.
    Tries the fork's Gemma3ChatHandler first (correct chat template), then
    falls back to the official package's Llava15ChatHandler (see module
    docstring's HONESTY NOTE) - either way this never raises.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        return None, f"llama-cpp-python not available: {exc}"

    handler = None
    handler_desc = None
    try:
        from llama_cpp.llama_chat_format import Gemma3ChatHandler
        handler = Gemma3ChatHandler(clip_model_path=MEDGEMMA_PROJECTOR_PATH, verbose=False)
        handler_desc = "Gemma3 chat handler"
    except ImportError:
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            handler = Llava15ChatHandler(clip_model_path=MEDGEMMA_PROJECTOR_PATH, verbose=False)
            handler_desc = "LLaVA-1.5 compatibility chat handler - experimental fallback"
        except ImportError as exc:
            return None, (
                f"No compatible multimodal chat handler found in llama-cpp-python ({exc}). Install "
                "the fork for full support: pip install "
                '"llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"'
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to initialise MedGemma chat handler: {exc}"

    try:
        instance = Llama(
            model_path=MEDGEMMA_MODEL_PATH,
            chat_handler=handler,
            n_ctx=MEDGEMMA_N_CTX,
            n_threads=MEDGEMMA_N_THREADS,
            n_batch=MEDGEMMA_N_BATCH,
            n_gpu_layers=0,  # CPU-only by design, per deployment constraint - see CUDA_VISIBLE_DEVICES above too
            use_mmap=True,
            verbose=False,
        )
        logger.info(
            "MedGemma-4B-IT loaded for CPU-only inference (n_ctx=%d, n_threads=%d, n_batch=%d, "
            "n_gpu_layers=0, %s).", MEDGEMMA_N_CTX, MEDGEMMA_N_THREADS, MEDGEMMA_N_BATCH, handler_desc,
        )
        return instance, f"MedGemma-4B-IT (Q4_K_M, local CPU inference, {handler_desc})"
    except Exception as exc:  # noqa: BLE001 - must never crash the app
        return None, (
            f"Failed to load MedGemma model ({exc}). This model/handler combination is best-effort "
            "- see triage_engine/vision.py's HONESTY NOTE."
        )


def _try_load_moondream():
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MoondreamChatHandler
    except ImportError as exc:
        return None, f"llama-cpp-python (with vision/chat-format support) not available: {exc}"

    try:
        chat_handler = MoondreamChatHandler(clip_model_path=VISION_PROJECTOR_PATH, verbose=False)
        instance = Llama(
            model_path=VISION_MODEL_PATH,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_threads=max(1, (os.cpu_count() or 2) - 1),
            n_gpu_layers=0,
            verbose=False,
        )
        return instance, VISION_MODEL_LABEL
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to load local vision model ({exc})."


def _get_active_vision_model():
    """Lazily load ONE vision model per process. Tries whichever model
    VISION_PREFER_MEDGEMMA says goes first, falling through to the other one
    if the preferred model's files are missing or it fails to actually load
    (e.g. no compatible chat handler installed) - so a partial/broken
    MedGemma install still degrades to moondream2 rather than to nothing.
    Returns (instance, kind, label) - instance is None (kind/label are None
    too) if nothing is available.
    """
    global _active_llm, _active_kind, _active_label, _load_attempted, _load_error
    if _active_llm is not None:
        return _active_llm, _active_kind, _active_label
    if _load_attempted:
        return None, None, None
    _load_attempted = True

    candidates = [
        ("medgemma", medgemma_files_present, _try_load_medgemma),
        ("moondream", model_files_present, _try_load_moondream),
    ]
    if not VISION_PREFER_MEDGEMMA:
        candidates.reverse()

    for kind, files_present, try_load in candidates:
        if not files_present():
            continue
        instance, result = try_load()
        if instance is not None:
            _active_llm, _active_kind, _active_label = instance, kind, result
            return _active_llm, _active_kind, _active_label
        _load_error = result if not _load_error else f"{_load_error}; {kind} also failed: {result}"

    if _load_error is None:
        _load_error = (
            f"No vision model files found in {MODELS_DIR}. Run `python download_model.py --vision` "
            "for basic image description, or `python download_model.py --medgemma` for AI-suggested "
            "possible findings from photos, then restart the app."
        )
    return None, None, None


def preload_async():
    """Kick off the vision model load on a background thread at process
    start, mirroring llm.preload_async() - so the multi-gigabyte weight load
    of whichever vision model is active (the slowest part by far, not the
    per-image inference itself) happens once, in the background, while the
    app is already serving requests, instead of stalling a user's first
    photo upload for however long the load takes.

    OFF by default (see VISION_PRELOAD_ENABLED above) - a no-op unless
    VISION_PRELOAD=1 is set, since eagerly loading a multi-gigabyte vision
    model for every process lifetime, whether or not any case ever has a
    photo, trades away memory headroom for a speed benefit most triage
    cases (text-only) never even use. Safe to call more than once - the
    _load_attempted flag inside _get_active_vision_model() makes this
    idempotent.
    """
    if not VISION_PRELOAD_ENABLED:
        return
    if not (medgemma_files_present() or model_files_present()):
        return
    threading.Thread(target=_get_active_vision_model, name="vision-preload", daemon=True).start()


def availability_status():
    """Human-readable status for the UI / /health endpoint."""
    instance, kind, label = _get_active_vision_model()
    if instance is not None:
        return True, f"AI image description active ({label})."
    return False, _load_error or "AI image description not available."


def findings_availability_status():
    """Like availability_status(), but specifically for the MedGemma-only
    'AI-suggested possible findings' feature (moondream2 cannot provide
    this - it has no medical-domain training).
    """
    instance, kind, label = _get_active_vision_model()
    if kind == "medgemma":
        return True, f"AI-suggested findings active ({label})."
    if medgemma_files_present() and not VISION_PREFER_MEDGEMMA:
        return False, (
            "AI-suggested possible findings requires MedGemma, which is downloaded but not active "
            "(moondream2 is preferred by default for speed/memory safety on constrained hardware - "
            "see vision.py's module docstring). Set VISION_PREFER_MEDGEMMA=1 and restart the app to "
            "enable it - expect a noticeably slower per-photo response in exchange."
        )
    return False, (
        "AI-suggested possible findings requires the MedGemma model, which is not currently "
        "loaded. " + (_load_error or "Run `python download_model.py --medgemma` to enable it.")
    )


# Both moondream2 and MedGemma's vision encoders resize whatever they're given down to a fixed
# internal resolution anyway (MedGemma: 896x896, observed directly in a real run's llama.cpp log -
# "clip_encode: ... nx=896, ny=896"). Sending a full-resolution phone photo (often several thousand
# pixels per side) buys nothing - the model never sees that extra detail - while costing real
# decode time and peak memory during preprocessing, on top of a needlessly large base64 payload.
# Downscaling first is a genuine speed AND memory optimization, not just a nice-to-have; capped via
# VISION_MAX_IMAGE_DIM so it can be tuned without touching code.
VISION_MAX_IMAGE_DIM = llm._env_int("VISION_MAX_IMAGE_DIM", 1024)


def _image_data_uri(image_path):
    """Base64 data-URI for the model call. Downscales large images first
    (see VISION_MAX_IMAGE_DIM above); falls back to the raw file bytes on
    any error (e.g. Pillow can't parse the format) so a preprocessing
    failure never blocks the upload - just means this one call skips the
    speed/memory optimization rather than failing outright.
    """
    import base64

    try:
        from PIL import Image
        import io
        with Image.open(image_path) as img:
            if max(img.size) > VISION_MAX_IMAGE_DIM:
                img.thumbnail((VISION_MAX_IMAGE_DIM, VISION_MAX_IMAGE_DIM), Image.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:  # noqa: BLE001 - fall back to the raw file rather than failing
        logger.warning("Image downscale before vision-model call failed (%s) - using original file.", exc)
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "jpeg"
        return f"data:image/{ext};base64,{b64}"


def describe_image(image_path):
    """Returns (description: str|None, model_label: str|None). Best-effort;
    never raises - any failure just means no AI description is attached
    and the photo is still available for manual reviewer inspection. Uses
    whichever vision model is active (moondream2 by default, or MedGemma if
    VISION_PREFER_MEDGEMMA=1 - see that flag's docs) - both understand this
    purely-descriptive, non-diagnostic prompt.
    """
    vllm, kind, label = _get_active_vision_model()
    if vllm is None:
        return None, None

    try:
        data_uri = _image_data_uri(image_path)
        result = vllm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": DESCRIBE_PROMPT},
                ],
            }],
            max_tokens=150,
            temperature=0.2,
        )
        text = result["choices"][0]["message"]["content"]
        text = llm._safety_filter(text)
        return (text, label) if text else (None, None)
    except Exception:  # noqa: BLE001
        return None, None


def _ensure_disclaimer(text):
    """Guarantee the mandatory non-diagnostic disclaimer is present, in code
    - not just requested in the prompt - since this output names possible
    conditions and the guarantee must not depend on model compliance alone.
    """
    if not text:
        return text
    if "not a diagnosis" in text.lower():
        return text
    return text.rstrip() + "\n\n" + MEDGEMMA_DISCLAIMER


def _parse_findings(raw_text):
    """Turns the model's '- possibly consistent with X' lines into a bounded,
    cleaned list of strings. Returns [] if nothing usable was found (e.g. the
    model correctly said the image wasn't applicable).
    """
    if not raw_text:
        return []
    findings = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("not applicable"):
            return []
        findings.append(cleaned[:MAX_FINDING_LEN])
        if len(findings) >= MAX_FINDINGS:
            break
    return findings


def suggest_possible_findings(image_path):
    """AI-SUGGESTED, hedged, disclaimed list of possible visible conditions
    for the reviewer to confirm - see the module docstring's SAFETY DESIGN
    section. Returns (findings: list[str], model_label: str) on success, or
    (None, None) if the MedGemma model isn't available/loaded, generation
    failed, or the output didn't pass the safety filter. This NEVER touches
    risk tiering and is always additional, reviewer-facing information.
    """
    vllm, kind, label = _get_active_vision_model()
    if vllm is None or kind != "medgemma":
        return None, None

    try:
        data_uri = _image_data_uri(image_path)
        result = vllm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": FINDINGS_PROMPT},
                ],
            }],
            max_tokens=220,
            temperature=0.2,
        )
        text = result["choices"][0]["message"]["content"]
        text = llm._safety_filter(text)
        if not text:
            return None, None
        findings = _parse_findings(text)
        if not findings:
            return None, None
        # Re-run the safety filter over the joined, disclaimer-guaranteed text
        # as a final defense-in-depth check before this is ever stored/shown.
        joined = _ensure_disclaimer("\n".join(f"- {f}" for f in findings))
        if not llm._safety_filter(joined):
            return None, None
        return findings, label
    except Exception:  # noqa: BLE001
        return None, None
