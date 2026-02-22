"""
utils/tts.py — Queued TTS using ElevenLabs with pyttsx3 fallback.

Set your API key via environment variable:
    export ELEVEN_API_KEY="sk-..."

Features:
  1. Queue-based TTS: all requests are processed sequentially
  2. No messages missed due to concurrent requests
  3. Throttled speak() for continuous updates (skips if text hasn't changed)
  4. speak_once() for important messages (always queued)
  5. Background worker thread handles all speech delivery
  6. Falls back to pyttsx3 on ElevenLabs errors
"""

import os
import io
import time
import threading
import queue as queue_module

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

# Try to import edge-tts as fallback TTS engine (thread-safe, no API key needed)
try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False

# Async helper for edge-tts (runs in thread pool to avoid blocking)
import asyncio
try:
    from concurrent.futures import ThreadPoolExecutor
    _EXECUTOR = ThreadPoolExecutor(max_workers=1)
except ImportError:
    _EXECUTOR = None

# ElevenLabs settings — tweak to taste
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "AeRdCCKzvd23BpJoofzx")
ELEVEN_MODEL    = "eleven_turbo_v2"   # lowest-latency model (~300 ms)

class TTSEngine:
    def __init__(self):
        self._queue = queue_module.Queue()
        self._last_text: str  = ""
        self._last_time: float = 0.0
        self._stop_event = threading.Event()
        self._client = None
        
        print("[tts] ========== TTS Initialization ==========")
        
        if _ELEVEN_AVAILABLE:
            try:
                self._client = ElevenLabs(api_key=_ELEVEN_KEY)
                print(f"[tts] ElevenLabs ready  (voice={ELEVEN_VOICE_ID}, model={ELEVEN_MODEL})")
            except Exception as e:
                print(f"[tts] Failed to initialize ElevenLabs: {e}")
                self._client = None
        else:
            print("[tts] ElevenLabs not available (key missing or package not installed).")
        
        # Check edge-tts availability
        print("[tts] Checking edge-tts...")
        if _EDGE_TTS_AVAILABLE:
            print(f"[tts] edge-tts available (thread-safe, no API key needed)")
        else:
            print("[tts] edge-tts not available. Install with: pip install edge-tts")
        
        # Start background worker thread
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        print("[tts] Queue worker thread started")
        print("[tts] ========== TTS Initialization Complete ==========")

    def reset_throttle(self) -> None:
        """Reset throttle state. Call when mode changes to avoid blocking repeated announcements."""
        self._last_text = ""
        self._last_time = 0.0
        print("[tts] 🔄 Throttle reset")

    def speak(self, text: str) -> None:
        """Throttled speak — skips if text unchanged within THROTTLE_SECS.
        Non-blocking: queues request and returns immediately.
        
        Use for continuous updates (e.g., directions "left", "left", "left")
        Use speak_once() for important one-time announcements instead.
        """
        text = text.strip()
        if not text:
            return
        
        now = time.time()
        if text == self._last_text and (now - self._last_time) < THROTTLE_SECS:
            print(f"[tts] Throttled (same text within {THROTTLE_SECS}s): {text[:40]}...")
            return  # Skip throttled message
        
        self._last_text = text
        self._last_time = now
        queue_size = self._queue.qsize()
        self._queue.put(("speak", text))
        if queue_size > 3:
            print(f"[tts] Queue building up ({queue_size+1} items)")
        else:
            print(f"[tts] Queued speak ({queue_size+1} in queue): {text[:40]}...")

    def speak_once(self, text: str) -> None:
        """Important message — always queued, not throttled.
        Non-blocking: queues request and returns immediately.
        """
        text = text.strip()
        if text:
            queue_size = self._queue.qsize()
            self._queue.put(("speak_once", text))
            print(f"[tts] Queued speak_once ({queue_size+1} in queue): {text[:40]}...")

    def stop(self) -> None:
        """Signal worker to stop processing."""
        self._stop_event.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_queue(self) -> None:
        """Background worker: process TTS requests sequentially."""
        print("[tts] Queue worker thread RUNNING")
        iterations = 0
        while not self._stop_event.is_set():
            iterations += 1
            try:
                msg_type, text = self._queue.get(timeout=2.0)
                print(f"[tts] 📥 Queue item #{iterations}: ({msg_type}) '{text[:60]}...'")
                self._say(text)
                print(f"[tts] Item #{iterations} complete")
                time.sleep(0.3)
            except queue_module.Empty:
                if iterations % 20 == 0:
                    print(f"[tts] ⏳ Worker idle...")
                continue
            except Exception as e:
                print(f"[tts] Queue error: {e}")
        print("[tts] ⛔ Queue worker thread STOPPED")

    def _say(self, text: str) -> None:
        """Execute TTS via ElevenLabs or edge-tts fallback."""
        print(f"[tts] _say() called with: '{text}'")
        print(f"[tts]   ElevenLabs available: {_ELEVEN_AVAILABLE}, client ready: {self._client is not None}")
        print(f"[tts]   edge-tts available: {_EDGE_TTS_AVAILABLE}")
        
        if self._client and _ELEVEN_AVAILABLE:
            try:
                print(f"[tts] Attempting ElevenLabs...")
                audio_bytes = self._client.text_to_speech.convert(
                    voice_id=ELEVEN_VOICE_ID,
                    text=text,
                    model_id=ELEVEN_MODEL,
                )
                print(f"[tts] ElevenLabs returned {len(audio_bytes)} bytes")
                self._play_audio(audio_bytes)
                return
            except Exception as exc:
                print(f"[tts] ElevenLabs error ({type(exc).__name__}): {exc}")
                print(f"[tts] → Falling back to edge-tts...")
        
        print(f"[tts] Using edge-tts fallback")
        self._say_edge_tts(text)

    def _say_edge_tts(self, text: str) -> None:
        """Fallback TTS using edge-tts (cloud-based, thread-safe).
        
        edge-tts is Microsoft's TTS engine—high quality, no API key needed.
        It's async, so we run it via asyncio event loop.
        """
        if not _EDGE_TTS_AVAILABLE:
            print(f"[tts] edge-tts not available. Install with: pip install edge-tts")
            return
        
        try:
            print(f"[tts] 🔊 edge-tts generating speech for: '{text[:60]}...'")
            
            # Run async function in a new event loop (safe for worker thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_bytes = loop.run_until_complete(self._edge_tts_get_audio(text))
                if audio_bytes:
                    print(f"[tts] edge-tts generated {len(audio_bytes)} bytes")
                    self._play_audio(audio_bytes)
                else:
                    print(f"[tts] edge-tts generated no audio")
            finally:
                loop.close()
            
        except Exception as e:
            print(f"[tts] edge-tts error ({type(e).__name__}): {e}")
    
    async def _edge_tts_get_audio(self, text: str) -> bytes:
        """Async helper to get audio from edge-tts."""
        try:
            communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            return b"".join(audio_chunks)
        except Exception as e:
            print(f"[tts] edge-tts async error: {e}")
            return None

    def _play_audio(self, audio_data) -> None:
        """Play audio bytes using available backend. Handles both bytes and generators."""
        # Convert generator to bytes if needed
        if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
            try:
                audio_bytes = b"".join(audio_data)
                print(f"[tts] Converted audio stream to {len(audio_bytes)} bytes")
            except Exception as e:
                print(f"[tts] Failed to convert audio stream: {e}")
                return
        else:
            audio_bytes = audio_data
        
        if _PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                print(f"[tts] Playing via pydub: {len(audio_bytes)} bytes")
                pydub_play(audio)
                print(f"[tts] Pydub playback done")
                return
            except Exception as e:
                print(f"[tts] pydub playback failed: {e}")
        
        if _SOUNDDEVICE_AVAILABLE:
            try:
                import soundfile as sf
                import sounddevice as sd
                print(f"[tts] Playing via sounddevice: {len(audio_bytes)} bytes")
                with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                    sd.play(f.read(), f.samplerate)
                    sd.wait()
                print(f"[tts] Sounddevice playback done")
                return
            except Exception as e:
                print(f"[tts] sounddevice playback failed: {e}")
        
        # Fallback: save to temp file
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            print(f"[tts] Playing via afplay: {temp_path}")
            os.system(f"afplay '{temp_path}'")  # macOS
            os.remove(temp_path)
            print(f"[tts] afplay done")
        except Exception as e:
            print(f"[tts] No audio playback available: {e}")
