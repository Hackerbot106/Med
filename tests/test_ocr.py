"""
Tests for triage_engine/ocr.py's is_usable_ocr_text() - a small but
important guard added after a real, confirmed bug: run_ocr() itself
returns a human-readable placeholder string (e.g. "[No text could be
extracted from this file...]") when an uploaded image has no OCR-able
text at all, which is exactly what happens for any non-text photo (an
X-ray, a wound photo) run through OCR alongside the vision model. Before
this guard existed, that placeholder string was being fed straight into
symptom extraction (rules-based AND the AI model) as if it were real
report content - see tests/test_pipeline.py's
TestOcrPlaceholderTextIsFiltered for the end-to-end regression test of
that actual bug.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import ocr


class TestIsUsableOcrText(unittest.TestCase):
    def test_real_extracted_text_is_usable(self):
        self.assertTrue(ocr.is_usable_ocr_text("BP 140/90, Temp 101F, Hb 11.2 g/dL"))

    def test_no_text_placeholder_is_not_usable(self):
        self.assertFalse(ocr.is_usable_ocr_text(
            "[No text could be extracted from this file - image may be low quality; "
            "please attach for manual reviewer inspection.]"
        ))

    def test_ocr_failed_message_is_not_usable(self):
        self.assertFalse(ocr.is_usable_ocr_text("[OCR failed: some tesseract error]"))

    def test_ocr_unavailable_message_is_not_usable(self):
        self.assertFalse(ocr.is_usable_ocr_text(
            "[OCR unavailable: PDF support (poppler) not installed in this environment]"
        ))

    def test_none_and_empty_and_whitespace_are_not_usable(self):
        self.assertFalse(ocr.is_usable_ocr_text(None))
        self.assertFalse(ocr.is_usable_ocr_text(""))
        self.assertFalse(ocr.is_usable_ocr_text("   \n  "))

    def test_leading_whitespace_around_a_placeholder_is_still_caught(self):
        # run_ocr() shouldn't produce this, but the check should be robust to it regardless.
        self.assertFalse(ocr.is_usable_ocr_text("  [No text could be extracted from this file...]  "))


if __name__ == "__main__":
    unittest.main()
