# wheredamilk 🥛

> Real-time assistive vision — **find items**, **read labels**, and **get details** by speaking. AI-powered object detection, text recognition, and voice guidance.

[![Demo Video](http://img.youtube.com/vi/gTKes_XwjXY/0.jpg)](http://www.youtube.com/watch?v=gTKes_XwjXY "WheredaMilk?")
---

## 🎯 Modes

The app has **4 distinct modes**, each engineered for a specific task:

### **1. FIND Mode** 🔍
**Command:** `"find milk"` or `"find orange juice"`

**What it does:**
- Scans the scene for an object matching your query
- Uses **two-stage matching:**
  1. First tries YOLO object classes (fast, accurate for trained objects like "bottle", "cup")
  2. Falls back to OCR text matching (flexible, finds labeled items like "COCA-COLA", "JUICE")
- Once found, **locks onto the target** and tracks it silently (no spam)
- Announces location once: *"Found milk on your left!"*
- Continues tracking with visual feedback

**Best for:** Finding specific products in a cluttered scene

---

### **2. WHAT Mode** 🤔
**Command:** `"what is this"` or `"what does this say"`

**What it does:**
- Waits 2-3 seconds (gives you time to stabilize the camera on an object)
- Identifies the **object class** and **position** on screen
- Announces once: *"I see a bottle on your center."*
- Automatically returns to idle

**Best for:** Quick identification of objects you're pointing at

---

### **3. READ Mode** 📖
**Command:** `"read"` or `"read this"`

**What it does:**
- **Immediately** finds the largest non-person object
- Runs OCR to extract all visible text
- Announces only the text: *"The text reads: COCA-COLA CLASSIC"*
- Automatically returns to idle

**Best for:** Reading labels, barcodes, or any text on a single object

---

### **4. DETAILS Mode** ✨ (Powered by Google Gemini)
**Command:** `"tell me more"` or `"tell me more about this product"`

**What it does:**
- Sends current frame to **Google Gemini Vision AI**
- Gets **detailed product analysis:** brand, ingredients (if food), use, weight, nutritional info
- Shows loading spinner while analyzing (2-5 seconds)
- Reads the full analysis aloud: *"This is Coca-Cola Classic, a carbonated soft drink from The Coca-Cola Company..."*
- Automatically returns to idle

**Setup required:** 
```bash
export GEMINI_API_KEY=your_api_key
```

Get your free API key from: https://ai.google.dev/

**Best for:** Deep product understanding, ingredient checking, brand verification

---

## Architecture

```
Voice Command ("find milk")
        ↓
SpeechListener (background thread)   utils/speech.py
        ↓
Webcam Frame (OpenCV 640×480)
        ↓
YOLOv8n Detection                    vision/yolo.py
        ↓
Mode Handler                         logic/modes.py
├── FindModeHandler
│   ├─ Match YOLO classes
│   └─ Fallback to OCR
├── WhatModeHandler
│   ├─ Wait 2-3s
│   └─ Identify & position
├── ReadModeHandler
│   └─ Extract text
└── DetailsModeHandler
    └─ Gemini Vision API
        ↓
EasyOCR Text Recognition             vision/ocr.py
        ↓
IoU Tracker                           logic/tracker.py
        ↓
Keyword Matching                      logic/match.py
        ↓
ElevenLabs/edge-tts 🎙️               utils/tts.py
```

---

## Project Structure

```
wheredamilk/
├── .env                 ← API keys (gitignored)
├── main.py              ← voice-controlled webcam loop
├── app.py               ← Flask REST API (optional)
├── requirements.txt
├── README.md
│
├── vision/
│   ├── yolo.py          ← YOLOv8n detector
│   ├── ocr.py           ← EasyOCR wrapper
│   └── gemini.py        ← Google Gemini Vision API
│
├── logic/
│   ├── match.py         ← keyword matching
│   ├── tracker.py       ← IoU single-target tracker
│   └── modes.py         ← Mode handlers (FIND, WHAT, READ, DETAILS)
│
└── utils/
    ├── tts.py           ← ElevenLabs + edge-tts (throttled)
    └── speech.py        ← continuous mic listener
```

**Key files:**
- `main.py` — orchestrates all modes, ~200 lines vs ~450 before refactor
- `logic/modes.py` — encapsulates each mode's logic in handler classes (NEW)
- `vision/gemini.py` — Gemini Vision API integration (token-optimized)
- `utils/tts.py` — queue-based thread-safe TTS with fallbacks

---

## Installation

```bash
cd /Users/balachandrads/Desktop/Projects/wheredamilk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS mic support (optional)
brew install portaudio && pip install pyaudio

# EasyOCR weights (~70 MB) and MiDaS weights (~400 MB) download automatically on first run
```

> **Note:** EasyOCR may take 30-60 seconds to initialize on first run as it downloads the model.

---

## Text Recognition (EasyOCR) ✨

**EasyOCR** provides high-quality text recognition comparable to Google ML Kit. No setup needed — works automatically.

> EasyOCR downloads its model (~70 MB) automatically on first run. This may take 30-60 seconds.

---

## Text-to-Speech (TTS) 🎙️

### Primary: ElevenLabs (Optional)

For premium natural voices via **ElevenLabs**:

1. Sign up at **[elevenlabs.io](https://elevenlabs.io)** (free — 10,000 chars/month)
2. Go to **Settings → API Keys** → create and copy your key
3. Add to `.env` in the project root:

```bash
ELEVEN_API_KEY=sk_your_key_here
ELEVEN_VOICE_ID=AeRdCCKzvd23BpJoofzx  # optional — Rachel is default
```

> The `.env` file is gitignored and **never pushed to GitHub**.

### Fallback: edge-tts (Built-in) ✨ 

If **no ElevenLabs key is set**, the app automatically falls back to **edge-tts** — Microsoft Edge's high-quality voices. **No API key needed!** Just works.

- **Fallback chain:** ElevenLabs → edge-tts → afplay (system speaker)
- **Best for:** Users without API keys, offline environments (sort of)
- **Quality:** Comparable to Google ML Kit voices

---

## Usage

### Run the app

```bash
python main.py
```

### Voice Commands

| Say | Mode | What Happens |
|---|---|---|
| `"find milk"` | **FIND** | Scans for "milk" (YOLO class OR text), locks on, tracks silently |
| `"find orange juice"` | **FIND** | Works for any item name with spaces |
| `"what is this"` | **WHAT** | Waits 2-3s, identifies object, announces position once |
| `"what does this say"` | **WHAT** | Same as above |
| `"read"` | **READ** | Immediately OCRs largest object, reads text once |
| `"read this"` | **READ** | Same as above |
| `"tell me more"` | **DETAILS** | Sends to Gemini, gets detailed product info |
| `"tell me more about this product"` | **DETAILS** | Same as above |
| `"stop"` / `"cancel"` | Any | Returns to idle, stops current mode |
| `"quit"` / `"exit"` | Any | Closes app |

**Press `q` in the OpenCV window to also quit.**

---

## Configuration

### Google Gemini API (for DETAILS mode)

1. Go to: https://ai.google.dev/
2. Sign in with Google account
3. Create API key (free tier: 60 requests/minute)
4. Add to `.env`:
   ```bash
   GEMINI_API_KEY=your_key_here
   ```

### ElevenLabs TTS (optional premium voice)

1. Go to: https://elevenlabs.io (free: 10k chars/month)
2. Create account, get API key
3. Add to `.env`:
   ```bash
   ELEVEN_API_KEY=sk_your_key_here
   ELEVEN_VOICE_ID=AeRdCCKzvd23BpJoofzx  # default: Rachel
   ```

**Without ElevenLabs, the app uses edge-tts (Microsoft voices) — no key needed!**

---

### Flask API (optional)

```bash
python app.py
# Server at http://localhost:5000

# Find an item
curl -X POST http://localhost:5000/find \
  -d '{"query":"milk"}' \
  -H 'Content-Type: application/json'

# Read text
curl -X POST http://localhost:5000/read

# Get status
curl http://localhost:5000/status
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8n detection |
| `opencv-python` | Webcam + drawing |
| `easyocr` | Text recognition (ML Kit-quality) |
| `torch` | PyTorch (required by EasyOCR) |
| `transformers` + `timm` | MiDaS depth model |
| `elevenlabs` | 🎙️ Premium TTS |
| `edge-tts` | 🎙️ Fallback TTS (no API key) |
| `SpeechRecognition` | Mic voice commands |
| `python-dotenv` | `.env` key loading |
| `flask` | Optional REST API |

---

*"wheredamilk — real-time navigation and label reading for blind users using object detection, depth estimation, and natural voice guidance."*
