"""
One-time download of the local, offline, CPU-only model(s) used for AI
features (see triage_engine/llm.py and triage_engine/vision.py). Needs
internet access ONCE; after this completes, the whole app runs fully
offline.

Usage:
    python download_model.py                # downloads the default 1.5B text model (~1 GB)
    python download_model.py --size small    # smaller/faster 0.5B text model (~400 MB)
    python download_model.py --size large    # higher quality 3.8B text model (~2.3 GB)
    python download_model.py --vision        # ALSO downloads the general-purpose vision model (~2 GB total)
    python download_model.py --medgemma      # ALSO downloads MedGemma-4B for AI-suggested image findings (~4.2 GB)
    python download_model.py --indic-translate  # ALSO downloads IndicTrans2 for dedicated Hindi/Odia translation

Indic translation note: Qwen2.5 (the default text model above) has weak,
unreliable Odia coverage - it isn't one of its officially supported
languages, and this project's own quality checks were built specifically
because it can drift into the wrong script (observed: Bengali instead of
Odia) or repeat itself. `--indic-translate` downloads IndicTrans2
(AI4Bharat, MIT licensed - the *license* is permissive, but read on), a
dedicated English->Indic machine-translation model purpose-built for
exactly this, used in preference to Qwen2.5 for translating the AI summary
once downloaded (see triage_engine/indic_translate.py). It needs extra
Python packages beyond this project's other AI features:
    pip install transformers torch IndicTransToolkit
IMPORTANT - this model's Hugging Face repo is GATED, exactly like MedGemma
above, even though its license is MIT: unauthenticated downloads are
refused. If the download below fails, visit
https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M, log in,
click "Agree and access repository", create a read-token at
https://huggingface.co/settings/tokens, then re-run with:
    HF_TOKEN=hf_xxx python download_model.py --indic-translate
The app works fine without any of this - it falls back to Qwen2.5's
translation, then to showing the English summary, exactly as before -
which is exactly the state a real deployment ends up in if this gating
requirement isn't noticed (see indic_translate.py's HONESTY NOTE).

MedGemma note: MedGemma sits under Google's "Health AI Developer Foundations"
terms of use. If the anonymous download below returns a 401/403, the files
require you to accept those terms on Hugging Face first and authenticate the
download with a token:
    1. Visit https://huggingface.co/unsloth/medgemma-4b-it-GGUF and accept
       any terms shown (a free Hugging Face account may be required).
    2. Create a read-token at https://huggingface.co/settings/tokens
    3. Re-run with the token available, e.g.:
         HF_TOKEN=hf_xxx python download_model.py --medgemma
The app works fine without MedGemma - the "AI-suggested possible findings"
feature simply stays unavailable (photo upload, OCR, and the general-purpose
AI image description via --vision all keep working) until it's present.

If your machine/network cannot complete a large download, or a URL below
has moved (model hosting pages do change filenames over time), download
the .gguf file manually from the linked Hugging Face page and save it into
the models/ folder next to this script with the exact filename printed
below - the app checks for that exact filename.
"""
import argparse
import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# (repo, filename, approx size) - official Qwen2.5-Instruct GGUF quantizations.
MODEL_CHOICES = {
    "small": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "~400 MB"),
    "default": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "~1.0 GB"),
    "large": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", "~2.1 GB"),
}

# Optional vision model (image understanding - see triage_engine/vision.py).
# CORRECTED after a real download failure: the original repo/filename pairing here
# (vikhyatk/moondream2, guessed GGUF filenames) does not exist - vikhyatk/moondream2 only
# publishes raw .safetensors weights, no GGUF at all, so every download attempt 404'd. Verified
# by actually listing the Hugging Face repo. The real, quantized GGUF conversion (the one
# llama-cpp-python's MoondreamChatHandler is built against) lives in cjpais/moondream2-llamafile
# instead. Q5_K is used here (not Q4, which this repo doesn't offer) as the smallest/fastest text
# model file available (~1.06 GB); the mmproj (vision projector) is only published at F16
# (~910 MB) - there is no smaller quantization of it. Combined (~2 GB) this is still well under
# half of MedGemma's ~4.2 GB footprint - see vision.py's module docstring for why that difference
# matters on CPU-only/memory-constrained hardware. For slightly better quality at a modest size
# cost, swap in "moondream2-050824-q8.gguf" (~1.51 GB) instead of the q5k file below.
VISION_FILES = [
    ("cjpais/moondream2-llamafile", "moondream2-050824-q5k.gguf", "~1.06 GB"),
    ("cjpais/moondream2-llamafile", "moondream2-mmproj-050824-f16.gguf", "~910 MB"),
]

# MedGemma-4B-IT: medical-domain vision-language model used for the
# "AI-suggested possible findings from a patient photo" feature (see
# triage_engine/vision.py). Gated under Google's "Health AI Developer
# Foundations" terms - see the module docstring above for the HF_TOKEN flow
# if the anonymous download below is rejected.
MEDGEMMA_FILES = [
    ("unsloth/medgemma-4b-it-GGUF", "medgemma-4b-it-Q4_K_M.gguf", "~2.5 GB"),
    ("unsloth/medgemma-4b-it-GGUF", "mmproj-BF16.gguf", "~1.7 GB"),
]

# IndicTrans2 (distilled, dedicated English->Indic translation - see
# triage_engine/indic_translate.py). This is a full HF `transformers` model
# repo (multiple files: weights, tokenizer, config), not a single GGUF file,
# so it's fetched with huggingface_hub.snapshot_download instead of the
# single-file _download_one() helper used for every other model above.
INDIC_TRANSLATE_REPO = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_TRANSLATE_DIRNAME = "indictrans2-en-indic-dist-200M"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:3d}%  ({mb_done:,.0f} MB / {mb_total:,.0f} MB)")
        sys.stdout.flush()


def _download_one(repo, filename, approx_size, optional=False):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    dest_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(dest_path):
        print(f"Already downloaded: {dest_path}")
        return True

    print(f"Downloading {filename} ({approx_size}) from Hugging Face...")
    print(f"URL: {url}")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        if hf_token:
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {hf_token}"})
            with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 1024 * 1024
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    out_file.write(block)
                    downloaded += len(block)
                    _progress(downloaded // block_size or 1, block_size, total_size)
        else:
            urllib.request.urlretrieve(url, dest_path, _progress)
        print(f"\n  Done -> {dest_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Download failed: {exc}")
        if "401" in str(exc) or "403" in str(exc) or "Forbidden" in str(exc):
            print(
                f"  This looks like a gated/authentication error. Visit https://huggingface.co/{repo} "
                "to accept any terms shown, create a token at https://huggingface.co/settings/tokens, "
                "and re-run with HF_TOKEN=hf_xxx set."
            )
        print(
            f"  Or download manually from https://huggingface.co/{repo} and save as:\n  {dest_path}"
        )
        if os.path.exists(dest_path):
            os.remove(dest_path)
        if not optional:
            print(
                "\n  The app works fine without this model too - it will simply fall back to the "
                "deterministic, rules-based summary until the model is present."
            )
        return False


def _download_indic_translate_model():
    dest_dir = os.path.join(MODELS_DIR, INDIC_TRANSLATE_DIRNAME)
    if os.path.isdir(dest_dir) and os.path.exists(os.path.join(dest_dir, "config.json")):
        print(f"Already downloaded: {dest_dir}")
        return True

    print(f"Downloading IndicTrans2 (distilled, dedicated Hindi/Odia translation) to {dest_dir}...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "  Requires the `huggingface_hub` package (installed automatically with `transformers`).\n"
            "  Run: pip install transformers torch IndicTransToolkit\n  then re-run this command."
        )
        return False

    # CORRECTED after realizing this repo is gated (confirmed by listing its Hugging Face page,
    # which shows "You need to agree to share your contact information to access this model") -
    # despite the model itself being MIT licensed, unauthenticated access is refused, exactly like
    # MedGemma (see MEDGEMMA note above). This was previously undocumented and unauthenticated here,
    # which explains a real symptom: with no HF_TOKEN, this download silently 401/403s, IndicTrans2
    # never becomes available, and every translation silently falls through to Qwen2.5 - whose own
    # Odia coverage is weak enough that it often fails its own quality check and shows English
    # instead (see indic_translate.py's module docstring and README §11.6). Passing the token here,
    # the same way _download_one() already does for MedGemma, fixes this.
    hf_token = os.environ.get("HF_TOKEN")
    try:
        snapshot_download(repo_id=INDIC_TRANSLATE_REPO, local_dir=dest_dir, token=hf_token)
        print(f"  Done -> {dest_dir}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  Download failed: {exc}")
        if not hf_token or "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
            print(
                f"  This model is GATED on Hugging Face (confirmed - it requires accepting terms "
                f"and being authenticated, even though it's MIT licensed). Visit "
                f"https://huggingface.co/{INDIC_TRANSLATE_REPO}, log in, click 'Agree and access "
                "repository', then create a read-token at https://huggingface.co/settings/tokens "
                f"and re-run with:\n  HF_TOKEN=hf_xxx python download_model.py --indic-translate"
            )
        else:
            print(
                f"  Download manually from https://huggingface.co/{INDIC_TRANSLATE_REPO} and save "
                f"the repo's files into:\n  {dest_dir}"
            )
        return False


def main():
    parser = argparse.ArgumentParser(description="Download local GGUF model(s) for AI features.")
    parser.add_argument("--size", choices=MODEL_CHOICES.keys(), default="default",
                         help="Text model size (default: 'default' = Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--vision", action="store_true",
                         help="Also download the optional local vision model for AI image description "
                              "(experimental - see triage_engine/vision.py)")
    parser.add_argument("--vision-only", action="store_true",
                         help="Download ONLY the vision model, skip the text model")
    parser.add_argument("--medgemma", action="store_true",
                         help="Also download MedGemma-4B for AI-suggested possible findings from patient "
                              "photos (gated model - see the note near the top of this file for HF_TOKEN)")
    parser.add_argument("--medgemma-only", action="store_true",
                         help="Download ONLY MedGemma, skip the text model")
    parser.add_argument("--indic-translate", action="store_true",
                         help="Also download IndicTrans2 for dedicated, higher-quality Hindi/Odia "
                              "translation (needs extra packages - see the note near the top of "
                              "this file)")
    parser.add_argument("--indic-translate-only", action="store_true",
                         help="Download ONLY IndicTrans2, skip the text model")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("This is a ONE-TIME download - the app runs fully offline afterward.\n")

    ok = True
    if not args.vision_only and not args.medgemma_only and not args.indic_translate_only:
        repo, filename, approx_size = MODEL_CHOICES[args.size]
        if args.size != "default":
            print(
                f"NOTE: '--size {args.size}' downloads a different filename than the default "
                f"triage_engine/llm.py expects. Update MODEL_FILENAME there to match, or re-run "
                f"with the default size.\n"
            )
        ok = _download_one(repo, filename, approx_size) and ok
        print()

    if args.vision or args.vision_only:
        print(
            "Downloading moondream2 (image description) - this is now the DEFAULT active vision "
            "model (see triage_engine/vision.py's VISION_PREFER_MEDGEMMA). NOTE: an earlier version "
            "of this project pointed at the wrong Hugging Face repo for this model (vikhyatk/"
            "moondream2, which only publishes raw .safetensors weights, no GGUF) and every download "
            "attempt 404'd - this has been corrected to the real GGUF conversion repo, verified by "
            "listing its files directly, but still has not been exercised end-to-end against real "
            "hardware in the environment that maintains this project (no internet access there - "
            "see vision.py's module docstring). If this section still fails, the app still works "
            "fine, just without AI image description.\n"
        )
        for repo, filename, approx_size in VISION_FILES:
            ok = _download_one(repo, filename, approx_size, optional=True) and ok
            print()

    if args.medgemma or args.medgemma_only:
        print(
            "Downloading MedGemma-4B (medical-domain vision model, for AI-suggested possible "
            "findings from patient photos). This is a larger, gated download (~4.2 GB total) - see "
            "the note near the top of this file if it fails with an authentication error.\n"
        )
        for repo, filename, approx_size in MEDGEMMA_FILES:
            ok = _download_one(repo, filename, approx_size, optional=True) and ok
            print()

    if args.indic_translate or args.indic_translate_only:
        print(
            "Downloading IndicTrans2 (dedicated Hindi/Odia translation, ~1 GB) - needs `transformers`, "
            "`torch`, and `IndicTransToolkit` installed to actually run (see the note near the top of "
            "this file). The app falls back to Qwen2.5's translation, then English, without it.\n"
        )
        ok = _download_indic_translate_model() and ok
        print()

    if ok:
        print("All requested model(s) ready. Run the app normally - AI features activate automatically.")
    else:
        print("Some downloads did not complete - see notes above. The app still runs fine either way.")
        sys.exit(1)


if __name__ == "__main__":
    main()
