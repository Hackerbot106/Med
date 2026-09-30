"""
Unit tests for triage_engine/indic_translate.py (the dedicated IndicTrans2
Hindi/Odia translation path added because Qwen2.5's Odia coverage is weak -
see that module's docstring). Like the other AI-feature test files, these
run WITHOUT torch/transformers/IndicTransToolkit installed or any model
downloaded - they exercise the pure-Python gating, fallback, and
quality-check-reuse logic directly.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import indic_translate


class TestLanguageGating(unittest.TestCase):
    def test_only_hindi_and_odia_are_wired_up(self):
        self.assertEqual(set(indic_translate.LANG_CODES.keys()), {"hi", "or"})
        self.assertEqual(indic_translate.LANG_CODES["hi"], "hin_Deva")
        self.assertEqual(indic_translate.LANG_CODES["or"], "ory_Orya")

    def test_unsupported_language_returns_none_without_touching_the_model(self):
        with patch.object(indic_translate, "_get_model") as mock_get_model:
            result = indic_translate.translate_summary("Some text.", "fr")
            self.assertIsNone(result)
            mock_get_model.assert_not_called()

    def test_empty_or_none_text_returns_none_without_touching_the_model(self):
        with patch.object(indic_translate, "_get_model") as mock_get_model:
            self.assertIsNone(indic_translate.translate_summary("", "hi"))
            self.assertIsNone(indic_translate.translate_summary(None, "or"))
            self.assertIsNone(indic_translate.translate_summary("   ", "hi"))
            mock_get_model.assert_not_called()


class TestGracefulDegradationWithoutTorch(unittest.TestCase):
    """These confirm the module behaves exactly like every other optional AI
    feature in this project when its (heavier, PyTorch-based) dependencies
    or model files simply aren't present - which is the real state of this
    test environment, so these exercise the actual code path, not a mock.
    """

    def test_model_files_present_returns_bool(self):
        self.assertIsInstance(indic_translate.model_files_present(), bool)

    def test_model_available_never_raises(self):
        self.assertIsInstance(indic_translate.model_available(), bool)

    def test_availability_status_never_raises(self):
        available, message = indic_translate.availability_status()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)

    def test_translate_summary_returns_none_when_unavailable(self):
        # With no model files downloaded (true in this environment), this
        # must degrade to None rather than raising - the caller (app.py)
        # then falls back to llm.generate_ai_translation(), then English.
        result = indic_translate.translate_summary("Patient reports fever.", "or")
        self.assertIsNone(result)


class TestQualityChecksReuseLlmModule(unittest.TestCase):
    """translate_summary() must reject a degenerate or wrong-script result
    using the SAME checks llm.py's generalist translation path uses - these
    simulate a loaded model to exercise that reuse without needing torch.
    """

    def _with_fake_model(self, fake_generate_output):
        class FakeTokenizer:
            def __call__(self, *a, **k):
                return {"input_ids": [[1, 2, 3]]}

            def batch_decode(self, tokens, skip_special_tokens=True):
                return [fake_generate_output]

        class FakeProcessor:
            def preprocess_batch(self, sentences, src_lang, tgt_lang):
                return sentences

            def postprocess_batch(self, decoded, lang):
                return decoded

        class FakeModel:
            def generate(self, **kwargs):
                return [[1, 2, 3]]

        return FakeModel(), FakeTokenizer(), FakeProcessor()

    def test_degenerate_output_is_rejected(self):
        looping_text = "ଜ୍ୱର ଏବଂ କାଶ " * 10
        fake = self._with_fake_model(looping_text)
        with patch.object(indic_translate, "_get_model", return_value=fake):
            result = indic_translate.translate_summary("Patient has fever and cough.", "or")
        self.assertIsNone(result)

    def test_wrong_script_output_is_rejected(self):
        # Bengali text returned when Odia was requested - the exact real
        # failure mode this whole module exists to avoid for Qwen2.5.
        bengali_text = "রোগীর জ্বর এবং কাশি আছে।"
        fake = self._with_fake_model(bengali_text)
        with patch.object(indic_translate, "_get_model", return_value=fake):
            result = indic_translate.translate_summary("Patient has fever and cough.", "or")
        self.assertIsNone(result)

    def test_correct_script_output_passes(self):
        odia_text = "ରୋଗୀଙ୍କର ଜ୍ୱର ଏବଂ କାଶ ଅଛି।"
        fake = self._with_fake_model(odia_text)
        with patch.object(indic_translate, "_get_model", return_value=fake):
            result = indic_translate.translate_summary("Patient has fever and cough.", "or")
        self.assertEqual(result, odia_text)


if __name__ == "__main__":
    unittest.main()
