# wheredamilk 🥛

> Real-time assistive vision — **find items** and **read labels** by speaking, guided by AI-powered depth and voice guidance.

---

## ✅ What's Working

### Core Pipeline
- [x] **YOLOv8n object detection** — real-time, 640×480, every 2nd frame (`vision/yolo.py`)
- [x] **PaddleOCR text reading** — crops top-1/2 boxes by confidence, reads label text (`vision/ocr.py`)
- [x] **MiDaS monocular depth** — real depth from a single RGB webcam, no depth camera needed (`vision/depth.py`)
- [x] **Keyword matching** — case-insensitive substring, e.g. "milk" in "DairyPure Whole Milk" (`logic/match.py`)
- [x] **Spatial direction** — left/right/ahead from bbox centre + MiDaS depth (bbox-area fallback) (`logic/direction.py`)
- [x] **IoU tracker** — locks onto target, tracks across frames, handles short occlusions (`logic/tracker.py`)
- [x] **ElevenLabs TTS** 🎙️ — natural, human-quality voice via `eleven_turbo_v2` model (`utils/tts.py`)
- [x] **Throttled speech** — speaks only on direction change or every ~1s (no spam)
- [x] **Continuous mic listener** — background thread, always listening (`utils/speech.py`)
- [x] **Voice commands** — "find milk", "read", "stop", "quit"
- [x] **Find mode** — YOLO → OCR top boxes → match → lock → track → speak directions continuously
- [x] **Read mode** — OCR largest box → speak label text once
- [x] **`main.py`** — fully voice-controlled webcam loop, OpenCV overlay
- [x] **`app.py`** — optional Flask REST API (`/find`, `/read`, `/status`)

---

## 🔜 Needs Work

- [ ] **End-to-end live testing** — run `main.py` with webcam, verify full pipeline
- [ ] **Re-lock after occlusion** — if tracker loses target entirely, re-trigger OCR search
- [ ] **Multi-target disambiguation** — two matching items visible → pick closer one via MiDaS
- [ ] **Confidence-gated OCR** — skip OCR if YOLO confidence < threshold
- [ ] **Vertical guidance** — "look higher / lower / bottom shelf"
- [ ] **iOS / Android app** → calls Flask `/find` and `/read`

---

## Architecture

```
Voice Command ("find milk")
        ↓
Mic Listener (background thread)    utils/speech.py
        ↓
Webcam (OpenCV 640×480)
        ↓
YOLOv8n — detect objects            vision/yolo.py
        ↓
MiDaS — estimate depth              vision/depth.py
        ↓
PaddleOCR — read text on crop       vision/ocr.py
        ↓
Keyword match                       logic/match.py
        ↓
IoU Tracker — lock target           logic/tracker.py
        ↓
Direction (left/right + depth)      logic/direction.py
        ↓
ElevenLabs TTS 🎙️ (throttled)       utils/tts.py
```

---

## Project Structure

```
wheredamilk/
├── .env                 ← API keys (gitignored, never pushed)
├── main.py              ← voice-controlled webcam loop
├── app.py               ← Flask REST API (optional)
├── requirements.txt
├── README.md
│
├── vision/
│   ├── yolo.py          ← YOLOv8n detector
│   ├── ocr.py           ← PaddleOCR wrapper
│   └── depth.py         ← MiDaS monocular depth
│
├── logic/
│   ├── match.py         ← keyword matching
│   ├── direction.py     ← left/right/ahead + real depth
│   └── tracker.py       ← IoU single-target tracker
│
└── utils/
    ├── tts.py           ← ElevenLabs TTS (throttled)
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

# MiDaS weights (~400 MB) download automatically on first run
```

---

## ElevenLabs Setup 🎙️

1. Sign up at **[elevenlabs.io](https://elevenlabs.io)** (free — 10,000 chars/month)
2. Go to **Settings → API Keys** → create and copy your key
3. Add to `.env` in the project root:

```bash
ELEVEN_API_KEY=sk_your_key_here
ELEVEN_VOICE_ID=Rachel        # optional — Rachel is default
```

> The `.env` file is gitignored and **never pushed to GitHub**.
> If no key is set, TTS will be disabled.

---

## Usage

### Run

```bash
python main.py
```

| Say | What happens |
|---|---|
| `"find milk"` | Scans scene, locks on milk, speaks live directions |
| `"find orange juice"` | Works for any item name |
| `"read"` / `"what is this"` | OCRs biggest box, speaks the label once |
| `"stop"` / `"cancel"` | Return to idle |
| `"quit"` / `"exit"` | Close app |

Press `q` in the OpenCV window to also quit.

### Flask API (optional)

```bash
python app.py

curl -X POST http://localhost:5000/find -d '{"query":"milk"}' -H 'Content-Type: application/json'
curl -X POST http://localhost:5000/read
curl http://localhost:5000/status
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8n detection |
| `opencv-python` | Webcam + drawing |
| `paddleocr` | Text recognition |
| `transformers` + `timm` | MiDaS depth model |
| `elevenlabs` | 🎙️ Natural TTS |
| `SpeechRecognition` | Mic voice commands |
| `python-dotenv` | `.env` key loading |
| `flask` | Optional REST API |

---

*"wheredamilk — real-time navigation and label reading for blind users using object detection, depth estimation, and natural voice guidance."*
