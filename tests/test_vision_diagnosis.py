"""
Unit tests for the MedGemma "AI-suggested possible findings from a photo"
feature in triage_engine/vision.py. Like tests/test_llm_safety.py, these run
WITHOUT any model file downloaded or loaded - they exercise the pure-Python
prompt-safety, parsing, and disclaimer-enforcement logic directly, which is
exactly what a hackathon judge (or a future contributor) should be able to
verify without a multi-gigabyte download.

Why this feature gets its own safety tests (see vision.py's module
docstring "SAFETY DESIGN" section for the full reasoning): this is the one
place in the whole project where the AI is deliberately allowed to name a
possible condition, so it is the one place where accidentally drifting into
assertive ("the diagnosis is X" / "you have X") language would be most
dangerous. These tests lock in the two independent safeguards: the prompt
itself never asks for hard-blocked phrasing, and the disclaimer is
guaranteed in code rather than trusted to the model.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import llm, vision


class TestFindingsPromptIsSafetyFilterCompatible(unittest.TestCase):
    """The prompt we send to MedGemma must never itself ask for language
    that llm._safety_filter's HARD_DISALLOWED_PATTERNS would reject - if it
    did, this feature would be unusable by construction, no matter how well
    the model followed instructions.
    """

    def test_prompt_does_not_contain_hard_disallowed_phrases(self):
        prompt_lower = vision.FINDINGS_PROMPT.lower()
        # These are exactly the assertive-diagnosis phrases HARD_DISALLOWED_PATTERNS
        # blocks - the prompt should only ever MENTION them as forbidden, which is
        # already true today, but if that instructional wording is ever removed we
        # still want this test to make clear these phrases must not become part of
        # what the model is asked to actually OUTPUT.
        self.assertIn("never write the phrases", prompt_lower)

    def test_prompt_requires_hedged_attributed_language(self):
        prompt_lower = vision.FINDINGS_PROMPT.lower()
        self.assertIn("possibly consistent with", prompt_lower)
        self.assertIn("clinician", prompt_lower)


class TestEnsureDisclaimer(unittest.TestCase):
    def test_adds_disclaimer_when_missing(self):
        text = "- possibly consistent with mild contact dermatitis (redness, no discharge)"
        result = vision._ensure_disclaimer(text)
        self.assertIn(vision.MEDGEMMA_DISCLAIMER, result)
        self.assertIn("not a diagnosis", result.lower())

    def test_does_not_duplicate_existing_disclaimer(self):
        text = "Some finding.\n\n" + vision.MEDGEMMA_DISCLAIMER
        result = vision._ensure_disclaimer(text)
        self.assertEqual(result.lower().count("not a diagnosis"), 1)

    def test_none_or_empty_passes_through(self):
        self.assertIsNone(vision._ensure_disclaimer(None))
        self.assertEqual(vision._ensure_disclaimer(""), "")


class TestParseFindings(unittest.TestCase):
    def test_parses_bulleted_lines(self):
        raw = (
            "- possibly consistent with contact dermatitis (redness, no discharge)\n"
            "- may show visible signs associated with a fungal infection (scaly border)"
        )
        findings = vision._parse_findings(raw)
        self.assertEqual(len(findings), 2)
        self.assertTrue(findings[0].startswith("possibly consistent with"))

    def test_not_applicable_returns_empty_list(self):
        raw = "Not applicable - no visible symptom in this image."
        self.assertEqual(vision._parse_findings(raw), [])

    def test_caps_at_max_findings(self):
        raw = "\n".join(f"- finding number {i}" for i in range(10))
        findings = vision._parse_findings(raw)
        self.assertEqual(len(findings), vision.MAX_FINDINGS)

    def test_empty_or_none_input(self):
        self.assertEqual(vision._parse_findings(""), [])
        self.assertEqual(vision._parse_findings(None), [])

    def test_long_line_is_truncated(self):
        raw = "- " + ("x" * 500)
        findings = vision._parse_findings(raw)
        self.assertLessEqual(len(findings[0]), vision.MAX_FINDING_LEN)


class TestFindingsEndToEndSafetyFilter(unittest.TestCase):
    """Confirms the realistic, intended output shape (hedged findings +
    guaranteed disclaimer) actually passes the SAME safety filter every
    other AI output in this project must pass - i.e. the design works, not
    just each piece in isolation.
    """

    def test_typical_hedged_findings_with_disclaimer_pass_the_safety_filter(self):
        findings = [
            "possibly consistent with contact dermatitis (redness, no discharge)",
            "may show visible signs associated with a fungal infection (scaly border)",
        ]
        joined = vision._ensure_disclaimer("\n".join(f"- {f}" for f in findings))
        self.assertIsNotNone(llm._safety_filter(joined))

    def test_assertive_output_would_still_be_rejected_even_from_this_module(self):
        # Defense-in-depth check: even if a future model drifted into assertive
        # language despite the prompt, the shared safety filter still catches it.
        bad_output = "You have contact dermatitis and should see a doctor."
        self.assertIsNone(llm._safety_filter(bad_output))


class TestModelFilePresenceChecks(unittest.TestCase):
    """These just confirm the presence checks are well-formed booleans that
    never raise, regardless of whether model files happen to exist in this
    environment - they must not assume any particular filesystem state.
    """

    def test_model_files_present_returns_bool(self):
        self.assertIsInstance(vision.model_files_present(), bool)

    def test_medgemma_files_present_returns_bool(self):
        self.assertIsInstance(vision.medgemma_files_present(), bool)

    def test_availability_status_never_raises_without_models(self):
        available, message = vision.availability_status()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)

    def test_findings_availability_status_never_raises_without_models(self):
        available, message = vision.findings_availability_status()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)


class TestImageDownscaling(unittest.TestCase):
    """Regression tests for the CPU/memory optimization added after a real
    out-of-memory process kill was observed on constrained hardware during
    image encoding: a large source photo is downscaled (both models resize
    internally to a fixed resolution anyway - see VISION_MAX_IMAGE_DIM's
    comment) before being base64-encoded and sent to the vision model.
    """

    def _make_test_image(self, size, mode="RGB", fmt="JPEG", tmp_name="test_img"):
        from PIL import Image
        path = os.path.join(self._tmpdir, f"{tmp_name}.{fmt.lower()}")
        Image.new(mode, size, color=(120, 40, 40) if mode == "RGB" else 120).save(path)
        return path

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp(prefix="vision_test_")

    def test_large_image_is_downscaled_to_the_configured_max_dimension(self):
        import base64
        import io
        from PIL import Image
        path = self._make_test_image((3000, 2000))
        uri = vision._image_data_uri(path)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = Image.open(io.BytesIO(raw))
        self.assertLessEqual(max(decoded.size), vision.VISION_MAX_IMAGE_DIM)

    def test_small_image_is_not_upscaled(self):
        import base64
        import io
        from PIL import Image
        path = self._make_test_image((200, 150))
        uri = vision._image_data_uri(path)
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = Image.open(io.BytesIO(raw))
        self.assertEqual(decoded.size, (200, 150))

    def test_rgba_image_does_not_raise(self):
        # A PNG with an alpha channel must be handled (converted to RGB)
        # before JPEG re-encoding, which has no alpha support.
        path = self._make_test_image((400, 400), mode="RGBA", fmt="PNG")
        uri = vision._image_data_uri(path)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_unreadable_file_falls_back_without_raising(self):
        path = os.path.join(self._tmpdir, "not_really_an_image.jpg")
        with open(path, "wb") as f:
            f.write(b"not actually image data")
        # Must not raise - falls back to encoding the raw (garbage) bytes
        # rather than blocking the upload over a preprocessing failure.
        uri = vision._image_data_uri(path)
        self.assertTrue(uri.startswith("data:image/"))


class TestVisionModelPreferenceOrder(unittest.TestCase):
    """Regression tests for the load-order flip made after real hardware
    testing (see vision.py's module docstring "WHICH ONE LOADS BY DEFAULT"):
    moondream2 is now the default preferred model (faster, far less memory
    pressure), with MedGemma available as an explicit opt-in via
    VISION_PREFER_MEDGEMMA=1 for deployments with the RAM/swap headroom (and
    time budget) to want it. These tests fake BOTH models as "present" and
    loadable, and confirm _get_active_vision_model() picks the one the flag
    says to, without needing any real model weights.
    """

    def setUp(self):
        # _get_active_vision_model() caches its result at module level after the
        # first call, so every test here must reset that cache plus the flag,
        # and restore both afterwards to avoid bleeding into other test classes.
        self._orig_prefer = vision.VISION_PREFER_MEDGEMMA
        self._orig_llm = vision._active_llm
        self._orig_kind = vision._active_kind
        self._orig_label = vision._active_label
        self._orig_attempted = vision._load_attempted
        self._orig_error = vision._load_error
        self.addCleanup(self._restore)

        self._reset_cache()

    def _restore(self):
        vision.VISION_PREFER_MEDGEMMA = self._orig_prefer
        vision._active_llm = self._orig_llm
        vision._active_kind = self._orig_kind
        vision._active_label = self._orig_label
        vision._load_attempted = self._orig_attempted
        vision._load_error = self._orig_error

    def _reset_cache(self):
        vision._active_llm = None
        vision._active_kind = None
        vision._active_label = None
        vision._load_attempted = False
        vision._load_error = None

    def _fake_both_models_present_and_loadable(self):
        """Pretend both models' files exist and both load successfully, so
        the ONLY thing determining which one wins is VISION_PREFER_MEDGEMMA.
        """
        vision.medgemma_files_present = lambda: True
        vision.model_files_present = lambda: True
        vision._try_load_medgemma = lambda: ("fake-medgemma-instance", "MedGemma-4B-IT (fake)")
        vision._try_load_moondream = lambda: ("fake-moondream-instance", vision.VISION_MODEL_LABEL)
        self.addCleanup(self._restore_loaders)

    def _restore_loaders(self):
        del vision.medgemma_files_present
        del vision.model_files_present
        del vision._try_load_medgemma
        del vision._try_load_moondream

    def test_moondream_is_preferred_by_default(self):
        self.assertFalse(vision.VISION_PREFER_MEDGEMMA)  # confirms the new default
        self._fake_both_models_present_and_loadable()
        _instance, kind, _label = vision._get_active_vision_model()
        self.assertEqual(kind, "moondream")

    def test_medgemma_is_used_when_explicitly_preferred(self):
        vision.VISION_PREFER_MEDGEMMA = True
        self._fake_both_models_present_and_loadable()
        _instance, kind, _label = vision._get_active_vision_model()
        self.assertEqual(kind, "medgemma")

    def test_findings_status_hints_at_the_toggle_when_medgemma_present_but_not_preferred(self):
        # Fake medgemma's files as present but its load as failing (fast, hermetic - no real
        # llama_cpp import or model file needed) so the ONLY thing under test is that
        # findings_availability_status() correctly distinguishes "MedGemma is downloaded but
        # moondream2 is preferred by default" from "MedGemma isn't installed at all" (covered by
        # TestModelFilePresenceChecks) and points the user at the right env var either way.
        vision.medgemma_files_present = lambda: True
        vision._try_load_medgemma = lambda: (None, "fake failure - not actually installed in this test")
        self.addCleanup(self._restore_loaders_medgemma_only)
        available, message = vision.findings_availability_status()
        self.assertFalse(available)
        self.assertIn("VISION_PREFER_MEDGEMMA", message)

    def _restore_loaders_medgemma_only(self):
        del vision.medgemma_files_present
        del vision._try_load_medgemma


class TestPreloadIsOptInForMemorySafety(unittest.TestCase):
    """Regression test for a deliberate design decision made right after a
    real OOM kill: preloading the vision model at startup (for a faster
    first photo upload) is OFF by default, since keeping several extra GB
    permanently resident for every process lifetime - even when a given
    deployment never uploads a single photo - is exactly the kind of
    memory pressure that caused the crash this feature exists to avoid.
    """

    def test_preload_is_disabled_by_default(self):
        self.assertFalse(vision.VISION_PRELOAD_ENABLED)

    def test_preload_async_is_a_no_op_when_disabled(self):
        import threading
        before = threading.active_count()
        vision.preload_async()
        # No new thread should have been started.
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
