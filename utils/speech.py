"""
utils/speech.py — Continuous microphone command listener.

Runs in a background thread.  The main loop calls `get_command()`
to fetch the latest spoken command (non-blocking, returns None if
nothing new).

Supported commands (examples):
    "find milk"    → ("find", "milk")
    "find orange juice" → ("find", "orange juice")
    "what is this"  → ("what", "")
    "what does this say" → ("what", "")
    "read"         → ("read", "")
    "tell me more about this product" → ("details", "")
    "tell me more"  → ("details", "")
    "stop"         → ("stop", "")
    "quit"         → ("quit", "")

Usage:
    listener = SpeechListener()
    listener.start()
    while True:
        cmd = listener.get_command()   # returns (action, query) or None
        ...
"""

import threading
import queue

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


def parse_command(text: str) -> tuple[str, str] | None:
    """
    Parse raw speech text into (action, query).

    Returns None if the text doesn't match any known command.
    """
    text = text.lower().strip()

    if text.startswith("find "):
        query = text[len("find "):].strip()
        return ("find", query) if query else None

    if text in ("what is this", "what does this say", "what is it"):
        return ("what", "")

    if text in ("read", "read this"):
        return ("read", "")

    if text in ("tell me more", "tell me more about this", "tell me more about this product", "more details", "more information"):
        return ("details", "")

    if text in ("stop", "cancel"):
        return ("stop", "")

    if text in ("quit", "exit"):
        return ("quit", "")

    return None


class SpeechListener:
    """
    Continuously listens to the microphone in a daemon thread and
    parses spoken commands into a queue.
    """

    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.6):
        self._q: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        if _SR_AVAILABLE:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = energy_threshold
            self._recognizer.pause_threshold = pause_threshold
            self._recognizer.dynamic_energy_threshold = True
        else:
            self._recognizer = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background listening thread."""
        if not _SR_AVAILABLE:
            print("[speech] WARNING: SpeechRecognition not installed.")
            print("[speech] Install:  pip install SpeechRecognition pyaudio")
            print("[speech] Falling back to keyboard — press Enter then type commands.")
            t = threading.Thread(target=self._keyboard_loop, daemon=True)
        else:
            t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()
        return self

    def get_command(self) -> tuple[str, str] | None:
        """
        Non-blocking.  Returns (action, query) if a new command is ready,
        otherwise None.
        """
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()

    # ── Internal loops ────────────────────────────────────────────────────────

    def _listen_loop(self):
        """Continuously listen and recognise speech."""
        print("[speech] Microphone listener active. Say 'find <item>' or 'read'.")
        while not self._stop_event.is_set():
            try:
                with sr.Microphone() as source:
                    # Quick ambient calibration on every loop to adapt to noise
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self._recognizer.listen(
                        source, timeout=3, phrase_time_limit=5
                    )
                try:
                    text = self._recognizer.recognize_google(audio)
                    print(f"[speech] Heard: \"{text}\"")
                    cmd = parse_command(text)
                    if cmd:
                        print(f"[speech] Command: {cmd}")
                        self._q.put(cmd)
                except sr.UnknownValueError:
                    pass  # couldn't understand — silently retry
                except sr.RequestError as e:
                    print(f"[speech] Google API error: {e}")
            except sr.WaitTimeoutError:
                pass  # no speech in this window — loop again
            except Exception as exc:
                print(f"[speech] Unexpected error: {exc}")

    def _keyboard_loop(self):
        """Fallback when SpeechRecognition is unavailable."""
        print("[speech] Keyboard mode: type commands ('find milk', 'read', 'quit')")
        while not self._stop_event.is_set():
            try:
                text = input("[cmd] > ").strip()
                cmd = parse_command(text)
                if cmd:
                    self._q.put(cmd)
                else:
                    print("[speech] Unknown command. Try: 'find milk' or 'read'")
            except (EOFError, KeyboardInterrupt):
                self._q.put(("quit", ""))
                break
