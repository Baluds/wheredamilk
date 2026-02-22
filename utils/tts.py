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
import io
import time
import threading

THROTTLE_SECS = 1.0

# ── ElevenLabs setup ──────────────────────────────────────────────────────────
_ELEVEN_KEY = os.environ.get("ELEVEN_API_KEY", "")

try:
    from elevenlabs.client import ElevenLabs
    _ELEVEN_AVAILABLE = bool(_ELEVEN_KEY)
except ImportError:
    _ELEVEN_AVAILABLE = False

# Try to import audio playback (fallback to write-to-file if unavailable)
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False
    try:
        import soundfile as sf
        import sounddevice as sd
        _SOUNDDEVICE_AVAILABLE = True
    except ImportError:
        _SOUNDDEVICE_AVAILABLE = False

# ElevenLabs settings — tweak to taste
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "AeRdCCKzvd23BpJoofzx")
ELEVEN_MODEL    = "eleven_turbo_v2"   # lowest-latency model (~300 ms)

class TTSEngine:
    def __init__(self):
        self._last_text: str  = ""
        self._last_time: float = 0.0
        self._lock = threading.Lock()

        if _ELEVEN_AVAILABLE:
            self._client = ElevenLabs(api_key=_ELEVEN_KEY)
            self.speak_once("ElevenLabs TTS engine initialized and ready.")
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
            # Get audio bytes from ElevenLabs
            audio_bytes = self._client.text_to_speech.convert(
                voice_id=ELEVEN_VOICE_ID,
                text=text,
                model_id=ELEVEN_MODEL,
            )
            
            self._play_audio(audio_bytes)
        except Exception as exc:
            print(f"[tts] ElevenLabs error: {exc}")
    
    def _play_audio(self, audio_data) -> None:
        """Play audio bytes using available backend. Handles both bytes and generators."""
        # Convert generator to bytes if needed
        if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
            try:
                audio_bytes = b"".join(audio_data)
            except Exception as e:
                print(f"[tts] Failed to convert audio stream: {e}")
                return
        else:
            audio_bytes = audio_data
        
        if _PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                pydub_play(audio)
                return
            except Exception as e:
                print(f"[tts] pydub playback failed: {e}")
        
        if _SOUNDDEVICE_AVAILABLE:
            try:
                import soundfile as sf
                import sounddevice as sd
                with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                    sd.play(f.read(), f.samplerate)
                    sd.wait()
                return
            except Exception as e:
                print(f"[tts] sounddevice playback failed: {e}")
        
        # Fallback: save to temp file
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            # Try to play with system command
            os.system(f"afplay '{temp_path}'")  # macOS
            os.remove(temp_path)
        except Exception as e:
            print(f"[tts] No audio playback available: {e}")
