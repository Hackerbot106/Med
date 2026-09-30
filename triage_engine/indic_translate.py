"""
Dedicated Indic-language machine translation (IndicTrans2, distilled 200M,
AI4Bharat) for translating the AI-generated English triage summary into
Hindi/Odia - added specifically because Qwen2.5 (the generalist chat model
in llm.py) has weak, unreliable Odia coverage.

WHY THIS EXISTS (read before touching llm.py's translation path instead):
Asking a small GENERALIST chat model to translate into Odia is exactly the
failure mode this project's own quality checks already had to be built to
catch (see llm.py's SCRIPT_RANGES / _script_match_ratio / _looks_degenerate,
and the docstring on generate_ai_translation): Odia is a very low-resource
language for a small multilingual chat model, which can silently drift into
a related-but-wrong script (observed during this project's own testing:
fluent-looking Bengali instead of Odia) or loop into a repeated phrase.
Qwen2.5's own model card does not list Odia among its officially supported
languages at all - Hindi translation from it is serviceable, Odia was
always the weaker case "papered over" by the quality checks silently
falling back to English rather than actually translating well.

IndicTrans2 is a machine translation model purpose-built, trained, and
benchmarked specifically for this: English -> all 22 scheduled Indian
languages, Odia (`ory_Orya`) included, from AI4Bharat (IIT Madras). It is
used here in PREFERENCE to the generalist chat model for this one task,
mirroring the same project-wide principle behind MedGemma being preferred
over a generalist vision model for medical image findings (see vision.py):
match the model to the job, don't ask one generalist model to do
everything a specialist would do better.

FALLBACK CHAIN (see app.py's triage_detail view): this module (preferred,
dedicated MT) -> llm.generate_ai_translation() (the Qwen-based generalist
path, kept as a fallback for when this heavier optional dependency isn't
installed) -> showing the original English summary (the final, always-safe
fallback, unchanged from before this module existed). Every layer degrades
gracefully; nothing here is a hard requirement for the app to run.

DEPENDENCY NOTE: unlike every other AI feature in this project (which only
needs llama-cpp-python), this one needs `transformers`, `torch` (the CPU
build is sufficient), and `IndicTransToolkit` - a heavier, PyTorch-based
stack, because no maintained GGUF/llama.cpp build of IndicTrans2 exists as
of when this was written. This is still fully offline/local/CPU-only at
inference time (see `_get_model` - torch is explicitly capped to CPU
threads, no GPU is used or required), just a larger one-time install and
download (~1-2 GB with the PyTorch runtime). If these packages, or the
model files, aren't present, `translate_summary()` returns None
immediately and the app behaves exactly as it did before this module
existed - it never becomes a hard dependency.

Model: ai4bharat/indictrans2-en-indic-dist-200M (distilled, ~200M
parameters, MIT licensed). See download_model.py's `--indic-translate` flag.

GATED REPO - CONFIRMED, READ THIS FIRST IF TRANSLATIONS LOOK LIKE QWEN2.5's
(mediocre Hindi, English fallback for Odia): despite the MIT license, this
model's Hugging Face repo requires logging in, clicking "Agree and access
repository", and authenticating the download with a token - the license
covers usage rights, not Hugging Face's own access gate on the repo itself.
This was originally undocumented and undone here (the downloader called
`snapshot_download()` with no token at all), which is a real, confirmed way
this feature silently never activates: the download 401/403s, this module's
`model_available()` stays False forever, and every translation falls
through to the Qwen2.5 path this module exists specifically to improve on
- Qwen2.5's own Hindi is serviceable-but-mediocre and its Odia routinely
fails the script-match/degeneracy checks below and falls back to English.
If that's what you're seeing, the fix is almost certainly the missing
token, not a bug in the translation logic itself: run
`HF_TOKEN=hf_xxx python download_model.py --indic-translate` after
accepting the repo's terms at
https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M (create a
token at https://huggingface.co/settings/tokens if you don't have one).

HONESTY NOTE: like vision.py's MedGemma/moondream2 integrations, this
module's "model present" code path was written and reasoned through
carefully but could not be run end-to-end against real downloaded weights
in the environment that built it (no internet access there). The
degradation path (packages/model absent -> clear message, no crash, app
falls back exactly as before) *is* exercised by this project's test suite
without needing the actual weights. Confirm this end-to-end on a machine
with internet access before relying on it in a live demo.
"""

import logging
import os

from . import llm  # reuse MODELS_DIR + the script-match/degeneracy quality checks

logger = logging.getLogger("triage_engine.indic_translate")

MODELS_DIR = llm.MODELS_DIR
MODEL_DIRNAME = "indictrans2-en-indic-dist-200M"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_DIRNAME)
MODEL_REPO = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_LABEL = "IndicTrans2 (distilled 200M, AI4Bharat - dedicated English->Indic MT)"

# FLORES-200 language codes IndicTrans2 expects. Only Hindi/Odia are wired
# up here because those are the only two non-English languages this app
# supports (see translator.SUPPORTED_LANGUAGES) - adding another supported
# UI language later just means adding its code here.
LANG_CODES = {"hi": "hin_Deva", "or": "ory_Orya"}
SRC_LANG_CODE = "eng_Latn"

_model = None
_tokenizer = None
_processor = None
_load_attempted = False
_load_error = None


def model_files_present():
    return os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json"))


def _get_model():
    """Lazily load the model once per process. Returns (model, tokenizer,
    processor), all None if the extra packages or model files aren't
    present - callers must handle that gracefully (they all do, below).
    """
    global _model, _tokenizer, _processor, _load_attempted, _load_error
    if _model is not None:
        return _model, _tokenizer, _processor
    if _load_attempted:
        return None, None, None
    _load_attempted = True

    if not model_files_present():
        _load_error = (
            f"IndicTrans2 model files not found in {MODEL_PATH}. This repo is GATED on Hugging "
            "Face (accept terms + a token, even though it's MIT licensed - see this module's "
            "GATED REPO note). Run `HF_TOKEN=hf_xxx python download_model.py --indic-translate` "
            "once (needs internet + a token from https://huggingface.co/settings/tokens after "
            "accepting terms at the model page) to enable dedicated, higher-quality Hindi/Odia "
            "translation of the AI summary - until then, Hindi/Odia translation falls back to "
            "Qwen2.5, whose weaker Odia coverage often shows English instead."
        )
        return None, None, None

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError as exc:
        _load_error = (
            f"Dedicated Indic translation needs extra packages that aren't installed ({exc}). Run "
            "`pip install transformers torch IndicTransToolkit` to enable it - the app works fine "
            "without them, falling back to the generalist AI model's translation (or English)."
        )
        return None, None, None

    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32,  # CPU-only, no GPU/flash-attn
        )
        model.eval()
        processor = IndicProcessor(inference=True)
        _model, _tokenizer, _processor = model, tokenizer, processor
        logger.info("IndicTrans2 (dedicated Hindi/Odia translation model) loaded from %s.", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - must never crash the app
        _load_error = f"Failed to load IndicTrans2 ({exc})."
        _model = None
    return _model, _tokenizer, _processor


def model_available():
    if _model is not None:
        return True
    model, _, _ = _get_model()
    return model is not None


def availability_status():
    if model_available():
        return True, f"Dedicated Hindi/Odia translation active ({MODEL_LABEL})."
    return False, _load_error or "Dedicated Indic translation not available."


def translate_summary(text, lang_code):
    """Best-effort English -> Hindi/Odia translation of the AI summary via
    IndicTrans2. Returns None (caller falls back - see module docstring's
    FALLBACK CHAIN) on any failure, unusable input, unsupported language,
    or a result that fails the SAME script-match / degeneracy quality
    checks already used for the generalist model's translation path in
    llm.py - reused here rather than re-implemented, since the failure
    modes (wrong script, repetition loops) are model-agnostic.
    """
    target = LANG_CODES.get(lang_code)
    if not target or not text or not text.strip():
        return None
    model, tokenizer, processor = _get_model()
    if model is None:
        return None

    try:
        batch = processor.preprocess_batch([text], src_lang=SRC_LANG_CODE, tgt_lang=target)
        inputs = tokenizer(batch, truncation=True, padding="longest", max_length=512, return_tensors="pt")
        # No explicit torch.no_grad() needed here - transformers' generate() already runs under
        # @torch.no_grad()/@torch.inference_mode() internally; this also keeps this function free
        # of a hard `import torch` at the call site (only _get_model() needs it, lazily, matching
        # the same lazy-import pattern llm.py and vision.py use for their own heavy dependencies).
        tokens = model.generate(**inputs, max_length=512, num_beams=4)
        decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        output = processor.postprocess_batch(decoded, lang=target)
        result = (output[0] if output else "").strip()
    except Exception as exc:  # noqa: BLE001 - a translation failure must never break the page
        logger.warning("IndicTrans2 translation call failed: %s", exc, exc_info=True)
        return None

    if not result:
        return None
    if llm._looks_degenerate(result):
        logger.warning(
            "IndicTrans2 output for lang=%s looked degenerate/repetitive. Raw output: %r",
            lang_code, result[:300],
        )
        return None
    ratio = llm._script_match_ratio(result, lang_code)
    if ratio < 0.6:
        logger.warning(
            "IndicTrans2 output for lang=%s did not use the expected script (only %.0f%% of "
            "letters matched). Raw output: %r", lang_code, ratio * 100, result[:300],
        )
        return None
    return result
