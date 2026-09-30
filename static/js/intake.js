// Client-side voice input using the browser's Web Speech API (no server
// dependency, no audio ever leaves the device except through the browser's
// own built-in recognition service). Falls back gracefully if unsupported.
//
// LANGUAGE NOTE (read before assuming this "just works" for every language):
// this project supports English/Hindi/Odia everywhere else (UI, translation,
// summaries - see triage_engine/translator.py's SUPPORTED_LANGUAGES), so
// voice input follows the SAME "preferred_language" selector on this page
// instead of being hardcoded to one language - a patient describing symptoms
// in Hindi or Odia should have their own words recognized directly, not be
// forced through English first. BUT: which languages the browser's built-in
// recognition engine actually supports is decided by the browser/OS/network
// service behind window.SpeechRecognition, NOT by this app - there is no
// official published list, and it can differ by browser and change over
// time. Hindi (hi-IN) is a major, broadly-supported language and should work
// reliably in Chrome. Odia (or-IN) is a lower-resource regional language and
// support is genuinely uncertain - it may work, may silently return nothing,
// or may fire a "language-not-supported" error depending on the browser. The
// code below requests the correct BCP-47 tag either way (so it works
// wherever the browser does support it) and surfaces a clear, honest message
// if the browser rejects the language, rather than failing silently - see
// onerror below. If Odia voice input doesn't work in your browser, that is a
// browser/OS limitation, not something this app's own code can route around;
// typing (already fully supported in all three languages) remains the
// reliable fallback.
(function () {
  const btn = document.getElementById("voice-btn");
  const status = document.getElementById("voice-status");
  const textarea = document.getElementById("symptom_text");
  const inputModeField = document.getElementById("input_mode");
  const languageSelect = document.getElementById("preferred_language");
  if (!btn) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    status.textContent = "Voice input not supported in this browser - please type symptoms.";
    btn.disabled = true;
    return;
  }

  // Maps this app's own language codes (triage_engine/translator.py's SUPPORTED_LANGUAGES) to
  // BCP-47 speech-recognition tags. "IN" region variants are used throughout since this project
  // is built for Indian healthcare facilities specifically.
  const SPEECH_LANG_TAGS = { en: "en-IN", hi: "hi-IN", or: "or-IN" };
  const DEFAULT_SPEECH_LANG = "en-IN";

  function currentSpeechLang() {
    const code = languageSelect ? languageSelect.value : "en";
    return SPEECH_LANG_TAGS[code] || DEFAULT_SPEECH_LANG;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = currentSpeechLang();

  // Recognition language is read fresh on every start() call (below), so switching the
  // "Preferred language" dropdown before pressing the mic picks it up automatically - no need to
  // re-select anything else.
  if (languageSelect) {
    languageSelect.addEventListener("change", function () {
      recognition.lang = currentSpeechLang();
    });
  }

  let listening = false;
  // The Web Speech API fires onerror THEN onend for the same failed session - without this flag,
  // onend's generic "Stopped listening." message would immediately overwrite the more useful
  // error message onerror just set (e.g. the "doesn't support Odia" message above), and the user
  // would never get to read it.
  let hadError = false;

  recognition.onresult = function (event) {
    let finalTranscript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript + " ";
      }
    }
    if (finalTranscript) {
      textarea.value = (textarea.value ? textarea.value.trim() + " " : "") + finalTranscript.trim();
      inputModeField.value = "voice";
    }
  };

  recognition.onerror = function (event) {
    hadError = true;
    if (event.error === "language-not-supported") {
      const langLabel = languageSelect
        ? (languageSelect.options[languageSelect.selectedIndex] || {}).text || recognition.lang
        : recognition.lang;
      status.textContent = (
        "This browser's voice input doesn't support " + langLabel + " - please type the " +
        "symptoms instead, or try a different browser/device."
      );
    } else if (event.error === "no-speech") {
      status.textContent = "No speech detected - try again, speaking clearly into the microphone.";
    } else {
      status.textContent = "Voice input error: " + event.error;
    }
  };

  recognition.onend = function () {
    listening = false;
    btn.classList.remove("recording");
    btn.textContent = "🎤 Start voice input";
    if (!hadError) {
      status.textContent = "Stopped listening.";
    }
  };

  btn.addEventListener("click", function () {
    if (listening) {
      recognition.stop();
      return;
    }
    try {
      hadError = false;
      recognition.lang = currentSpeechLang();  // pick up any language change since the last start()
      recognition.start();
      listening = true;
      btn.classList.add("recording");
      btn.textContent = "⏹ Stop voice input";
      status.textContent = "Listening (" + recognition.lang + ")... speak the patient's symptoms.";
    } catch (e) {
      status.textContent = "Could not start voice input: " + e.message;
    }
  });
})();
