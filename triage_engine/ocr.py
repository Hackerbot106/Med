"""
OCR pipeline for uploaded lab reports / prescriptions / referral slips.

Supports: PNG/JPG images directly via pytesseract, and PDF files by
rasterising each page with pdf2image (poppler) before OCR.

Privacy note: uploaded files are stored under data/uploaded_reports with a
randomised filename tied to the triage_id, not the patient's real name.
Demo deployments should purge this folder regularly (see README).
"""

import os
import uuid
import pytesseract
from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploaded_reports")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"}


def save_upload(file_storage, triage_id):
    """Save an uploaded Werkzeug FileStorage object with a randomised name."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file_storage.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    safe_name = f"{triage_id}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    file_storage.save(path)
    return path, safe_name


def run_ocr(file_path):
    """Return extracted text from an image or PDF file. Best-effort: OCR
    quality depends heavily on scan/photo quality, which is expected and
    surfaced to the reviewer rather than hidden.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text_chunks = []
    try:
        if ext == ".pdf":
            if not PDF_SUPPORT:
                return "[OCR unavailable: PDF support (poppler) not installed in this environment]"
            pages = convert_from_path(file_path, dpi=200)
            for page_image in pages:
                text_chunks.append(pytesseract.image_to_string(page_image))
        else:
            image = Image.open(file_path)
            text_chunks.append(pytesseract.image_to_string(image))
    except Exception as exc:  # noqa: BLE001 - surface any OCR failure to the UI
        return f"[OCR failed: {exc}]"

    combined = "\n".join(chunk.strip() for chunk in text_chunks if chunk and chunk.strip())
    return combined or "[No text could be extracted from this file - image may be low quality; please attach for manual reviewer inspection.]"


# Sentinel prefixes run_ocr() itself produces above when no real text was found or OCR failed -
# these strings are meant to be DISPLAYED to the reviewer (so they know to look at the photo
# manually), never treated as clinical content. A real, confirmed bug this guards against: every
# uploaded image runs through OCR regardless of what it actually is - so uploading a photo that
# was never a text document at all (e.g. an X-ray, a wound photo) makes run_ocr() correctly find
# no text and return one of these placeholder strings, but without this check that placeholder
# string was being fed straight into the AI/rules extraction as if it were "OCR text from an
# uploaded report" (see pipeline.run_triage_pipeline()) - producing confusing extraction output
# that looks like it came from nowhere, on every non-text photo upload, X-rays included.
_OCR_NON_CONTENT_PREFIXES = ("[No text could be extracted", "[OCR failed:", "[OCR unavailable:")


def is_usable_ocr_text(text):
    """True if `text` looks like real extracted text rather than one of run_ocr()'s own
    placeholder/error messages above. Callers that treat ocr_text as clinical content
    (extraction, AI prompts - see pipeline.run_triage_pipeline()) should check this first;
    callers that just display ocr_text to the reviewer should NOT filter it - the placeholder
    messages are useful, intentional information for a human, just not for a model.
    """
    if not text or not text.strip():
        return False
    return not text.strip().startswith(_OCR_NON_CONTENT_PREFIXES)


def basic_image_quality_note(file_path):
    """Very lightweight heuristic image-quality check (brightness / size)
    so the reviewer knows if a re-scan / better photo may be needed. This
    is NOT clinical image analysis and makes no diagnostic claim.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return None
    try:
        with Image.open(file_path) as img:
            grayscale = img.convert("L")
            width, height = grayscale.size
            pixels = list(grayscale.getdata())
            avg_brightness = sum(pixels) / len(pixels) if pixels else 0
        notes = []
        if width < 600 or height < 600:
            notes.append("low resolution image - text extraction may be unreliable")
        if avg_brightness < 60:
            notes.append("image appears very dark")
        elif avg_brightness > 235:
            notes.append("image appears washed out / overexposed")
        return "; ".join(notes) if notes else "image quality looks adequate"
    except Exception:
        return None
