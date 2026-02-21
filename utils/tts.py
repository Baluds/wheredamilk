"""
utils/tts.py — Throttled TTS using ElevenLabs.

Set your API key via environment variable:
    export ELEVEN_API_KEY="sk-..."

Speaks only when:
  1. The text has changed since last utterance, OR
  2. More than THROTTLE_SECS seconds have passed.

All speech runs in a background daemon thread — never blocks the webcam loop.
"""

import os
import time
import threading

THROTTLE_SECS = 1.0

# ── ElevenLabs setup ──────────────────────────────────────────────────────────
_ELEVEN_KEY = os.environ.get("ELEVEN_API_KEY", "")

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play
    _ELEVEN_AVAILABLE = bool(_ELEVEN_KEY)
except ImportError:
    _ELEVEN_AVAILABLE = False

# ElevenLabs settings — tweak to taste
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "Rachel")
ELEVEN_MODEL    = "eleven_turbo_v2"   # lowest-latency model (~300 ms)

class TTSEngine:
    def __init__(self):
        self._last_text: str  = ""
        self._last_time: float = 0.0
        self._lock = threading.Lock()

        if _ELEVEN_AVAILABLE:
            self._client = ElevenLabs(api_key=_ELEVEN_KEY)
            print(f"[tts] ElevenLabs ready  (voice={ELEVEN_VOICE_ID}, model={ELEVEN_MODEL})")
        else:
            print("[tts] WARNING: ElevenLabs not available (key missing or package not installed). No audio.")

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Throttled speak — non-blocking."""
        text = text.strip()
        if not text:
            return
        now = time.time()
        with self._lock:
            if text == self._last_text and (now - self._last_time) < THROTTLE_SECS:
                return
            self._last_text = text
            self._last_time = now
        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    def speak_once(self, text: str) -> None:
        """Speak unconditionally and block until finished (used for 'read' mode)."""
        text = text.strip()
        if text:
            self._say(text)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _say(self, text: str) -> None:
        if _ELEVEN_AVAILABLE:
            self._say_eleven(text)
        else:
            print(f"[tts] (no audio) {text}")

    def _say_eleven(self, text: str) -> None:
        try:
            audio = self._client.text_to_speech.convert(
                voice_id=ELEVEN_VOICE_ID,
                text=text,
                model_id=ELEVEN_MODEL,
            )
            play(audio)
        except Exception as exc:
            print(f"[tts] ElevenLabs error: {exc}")
