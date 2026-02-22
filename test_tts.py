#!/usr/bin/env python3
"""
Quick diagnostic test for TTS system
"""

import sys
import time

print("=" * 60)
print("TTS DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: pyttsx3 basic
print("\n[TEST 1] Testing pyttsx3 basic...")
try:
    import pyttsx3
    print("✅ pyttsx3 imported")
    
    engine = pyttsx3.init()
    print(f"✅ pyttsx3.init() successful")
    print(f"   Driver: {engine.driver_name}")
    print(f"   Rate: {engine.getProperty('rate')}")
    print(f"   Volume: {engine.getProperty('volume')}")
    print(f"   Voices: {len(engine.getProperty('voices'))} available")
    
    engine.setProperty('rate', 150)
    print("✅ pyttsx3 properties set")
    
    print("\n🔊 Speaking test phrase...")
    engine.say("Hello, this is a test")
    engine.runAndWait()
    print("✅ pyttsx3 test complete")
    
except Exception as e:
    print(f"❌ pyttsx3 test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: TTSEngine queue
print("\n[TEST 2] Testing TTSEngine...")
try:
    from utils.tts import TTSEngine
    print("✅ TTSEngine imported")
    
    tts = TTSEngine()
    time.sleep(1)  # Wait for init
    
    print("\n📤 Queuing test message...")
    tts.speak_once("Queue test message")
    
    print("⏳ Waiting for queue processing...")
    time.sleep(3)
    
    print("\n📤 Queuing another message...")
    tts.speak_once("Second test message")
    
    print("⏳ Waiting...")
    time.sleep(3)
    
    tts.stop()
    time.sleep(0.5)
    print("✅ TTSEngine test complete")
    
except Exception as e:
    print(f"❌ TTSEngine test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
