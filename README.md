# wheredamilk 🥛

> Real-time assistive vision — **find items** and **read labels** by speaking, guided by AI-powered depth and direction audio.

---

## ✅ What's Done

### Core Pipeline
- [x] **YOLOv8n object detection** — real-time, 640×480, every 2nd frame (`vision/yolo.py`)
- [x] **PaddleOCR text reading** — crops top-1/2 boxes by confidence, reads label text (`vision/ocr.py`)
- [x] **MiDaS monocular depth** — estimates real depth from single RGB webcam, no depth camera needed (`vision/depth.py`)
- [x] **Keyword matching** — case-insensitive substring, e.g. "milk" in "DairyPure Whole Milk" (`logic/match.py`)
- [x] **Spatial direction** — left/right/ahead from bbox centre + real MiDaS depth (bbox-area fallback) (`logic/direction.py`)
- [x] **IoU tracker** — locks onto target, tracks across frames, handles short occlusions (`logic/tracker.py`)

### Audio
- [x] **ElevenLabs TTS** — natural voice guidance via `eleven_turbo_v2` (`utils/tts.py`)
- [x] **pyttsx3 fallback** — offline TTS if ElevenLabs key not set
- [x] **Throttled speech** — speaks only on direction change or every ~1s (no spam)
- [x] **Voice command input** — continuous mic listener in background thread (`utils/speech.py`)
- [x] **Command parsing** — "find milk", "read", "stop", "quit"

### Modes
- [x] **Find mode** — YOLO → OCR top boxes → match → lock → track → speak directions continuously
- [x] **Read mode** — OCR largest box → speak label text once

### App
- [x] **`main.py`** — fully voice-controlled webcam loop, OpenCV overlay
- [x] **`app.py`** — optional Flask REST API (`/find`, `/read`, `/status`)

---

## 🔜 Not Yet Done / To-Do

### Accuracy & Robustness
- [ ] **Offline speech recognition** — currently uses Google Speech API (internet required); swap to Whisper or Vosk for offline use
- [ ] **Vertical guidance** — currently only left/right/forward; no up/down guidance (e.g. "look higher", "it's on the bottom shelf")
- [ ] **Multi-target disambiguation** — if two "milk" cartons are visible, pick the closer one using depth
- [ ] **Re-lock after occlusion** — if IoU tracker loses the target completely, re-trigger OCR search
- [ ] **Confidence-gated OCR** — skip OCR if YOLO confidence < threshold (reduce false positives)

### User Experience
- [ ] **Wake word** — say "hey milk" to activate, instead of always listening
- [ ] **Earpiece / headphone mode** — suppress OpenCV window, audio-only output for real device use
- [ ] **Battery / speed mode toggle** — skip MiDaS on low-power mode, use bbox-area only

### Platform
- [ ] **iOS / Android front-end** — connect to `app.py` Flask API from a mobile app
- [ ] **Raspberry Pi / edge deploy** — test on-device with Pi Camera Module

---

## Architecture

```
Webcam (OpenCV)
      ↓
YOLOv8n (detect objects)            vision/yolo.py
      ↓
Select top 1–2 boxes by confidence
      ↓
MiDaS Depth Estimator               vision/depth.py    ← NEW
      ↓
PaddleOCR (read text on crop)       vision/ocr.py
      ↓
Keyword match                       logic/match.py
      ↓
IoU Tracker (lock target)           logic/tracker.py
      ↓
Direction (left/right + depth)      logic/direction.py
      ↓
ElevenLabs TTS (throttled)          utils/tts.py
      ↑
Voice Commands (mic thread)         utils/speech.py
```

---

## Project Structure

```
wheredamilk/
├── main.py              ← voice-controlled webcam loop
├── app.py               ← Flask REST API (optional)
├── requirements.txt
├── README.md
│
├── vision/
│   ├── yolo.py          ← YOLOv8n detector
│   ├── ocr.py           ← PaddleOCR wrapper
│   └── depth.py         ← MiDaS monocular depth  ← NEW
│
├── logic/
│   ├── match.py         ← keyword matching
│   ├── direction.py     ← left/right/ahead + depth
│   └── tracker.py       ← IoU single-target tracker
│
└── utils/
    ├── tts.py           ← ElevenLabs + pyttsx3 fallback
    └── speech.py        ← continuous mic listener
```

---

## Installation

```bash
cd /Users/athulyaanil/wheredamilk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS mic support
brew install portaudio && pip install pyaudio

# MiDaS depth model (~400 MB, downloads on first run automatically)
# Nothing extra to install — handled by: pip install transformers timm torch
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ELEVEN_API_KEY` | ✅ For best voice | ElevenLabs API key — get at [elevenlabs.io](https://elevenlabs.io) |
| `ELEVEN_VOICE_ID` | Optional | Voice name (default: `Rachel`) |

```bash
export ELEVEN_API_KEY="sk-..."
export ELEVEN_VOICE_ID="Rachel"   # or Bella, Antoni, etc.
```

---

## Usage

### Standalone

```bash
python main.py
```

| Say | Action |
|---|---|
| `"find milk"` | Scans, locks on, speaks live directions until you reach it |
| `"find orange juice"` | Works for any item |
| `"read"` / `"what is this"` | OCR the biggest thing in view, speak label once |
| `"stop"` / `"cancel"` | Return to idle |
| `"quit"` / `"exit"` | Close app |

Press `q` in the OpenCV window to also quit.

### Flask API

```bash
python app.py

curl -X POST http://localhost:5000/find -d '{"query":"milk"}' -H 'Content-Type: application/json'
curl -X POST http://localhost:5000/read
curl http://localhost:5000/status
```

---

## Tech Stack

| Library | Purpose | Status |
|---|---|---|
| `ultralytics` | YOLOv8n detection | ✅ Done |
| `opencv-python` | Webcam + drawing | ✅ Done |
| `paddleocr` | Text recognition | ✅ Done |
| `transformers` + `timm` | MiDaS depth model | ✅ Done |
| `elevenlabs` | Natural TTS | ✅ Done |
| `pyttsx3` | Offline TTS fallback | ✅ Done |
| `SpeechRecognition` | Mic voice commands | ✅ Done |
| `flask` | Optional REST API | ✅ Done |

---

*"wheredamilk provides real-time navigation and label reading for blind users using object detection, depth estimation, and OCR."*
