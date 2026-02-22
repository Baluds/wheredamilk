"""
test_gemini.py — Test Google Gemini Vision API on webcam frames.

This script captures frames from your webcam and sends them to Google Gemini
for product identification and analysis.

Setup:
    1. Get API key from https://ai.google.dev/
    2. Add to .env: GEMINI_API_KEY=your_key_here
    3. Run: python test_gemini.py

Controls:
    SPACE  → Analyze current frame
    Q      → Identify what query (if set)
    T      → Test mode (auto-analyze every 5 frames)
    C      → Set custom query
    X      → Quit
"""

import sys
import cv2
import os

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vision.gemini import GeminiAnalyzer

# ── Constants ─────────────────────────────────────────────────────────────────
FRAME_W = 640
FRAME_H = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX


class GeminiTest:
    def __init__(self):
        self.analyzer = GeminiAnalyzer()
        self.test_mode = False
        self.test_counter = 0
        self.custom_query = ""
        self.last_analysis = None
    
    def draw_on_frame(self, frame):
        """Draw UI and results on frame."""
        display = frame.copy()
        
        # Status
        status = "Gemini Test — SPACE (analyze) | Q (query) | T (test mode) | C (custom query) | X (quit)"
        cv2.putText(display, status, (5, 20), FONT, 0.40, (200, 200, 200), 1)
        
        if self.test_mode:
            cv2.putText(display, "TEST MODE ON — auto-analyzing every 5 frames", 
                       (5, 45), FONT, 0.40, (0, 255, 255), 1)
        
        if self.custom_query:
            cv2.putText(display, f"Query: {self.custom_query[:50]}...", 
                       (5, 70), FONT, 0.40, (0, 255, 0), 1)
        
        if self.last_analysis:
            y_offset = 95
            if "products" in self.last_analysis:
                cv2.putText(display, f"Products: {len(self.last_analysis.get('products', []))} detected", 
                           (5, y_offset), FONT, 0.35, (100, 255, 100), 1)
                y_offset += 20
                for prod in self.last_analysis.get("products", [])[:3]:
                    text = f"  • {prod['name']}"
                    cv2.putText(display, text, (5, y_offset), FONT, 0.30, (100, 255, 100), 1)
                    y_offset += 15
        
        return display
    
    def analyze_frame_simple(self, frame):
        """Analyze frame and display results."""
        if not self.analyzer.available():
            print("[test] Gemini not available. Check your API key in .env")
            print("       GEMINI_API_KEY=your_key_here")
            return
        
        print("\n" + "="*60)
        print("ANALYZING FRAME...")
        print("="*60)
        
        result = self.analyzer.analyze_frame(frame, detailed=True)
        self.last_analysis = result
        
        if result.get("success"):
            print(f"\n✅ Analysis successful\n")
            
            # Products
            if result.get("products"):
                print(f"🛍️  PRODUCTS ({len(result['products'])}):")
                for prod in result["products"]:
                    print(f"  • {prod['name']}: {prod['description']}")
            else:
                print("🛍️  PRODUCTS: None detected")
            
            # Text
            if result.get("text"):
                print(f"\n📝 VISIBLE TEXT:\n  {result['text']}")
            
            # Description
            if result.get("description"):
                print(f"\n📋 DESCRIPTION:\n  {result['description']}")
        else:
            print(f"\n❌ Analysis failed: {result.get('error')}")
        
        print("\n" + "="*60 + "\n")
    
    def query_frame(self, frame, query: str):
        """Ask Gemini a specific question about the frame."""
        if not self.analyzer.available():
            print("[test] Gemini not available. Check your API key in .env")
            return
        
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        
        result = self.analyzer.identify_product(frame, query)
        
        if result.get("success"):
            print(f"\n✅ Response:\n\n{result['answer']}\n")
        else:
            print(f"\n❌ Query failed: {result.get('error')}")
        
        print("="*60 + "\n")
    
    def main(self):
        print("\n▶  Gemini Vision Test starting…\n")
        
        if not self.analyzer.available():
            print("❌ Gemini is not available!")
            print("\nTo use Gemini:")
            print("  1. Get API key from: https://ai.google.dev/")
            print("  2. Create .env file with: GEMINI_API_KEY=your_api_key")
            print("  3. Install: pip install google-generativeai")
            sys.exit(1)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("ERROR: Cannot open webcam.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        
        cv2.namedWindow("Gemini Test")
        
        print("═" * 60)
        print("CONTROLS:")
        print("  SPACE  → Analyze frame with Gemini")
        print("  Q      → Query: identify specific product")
        print("  T      → Toggle test mode (auto-analyze every 5 frames)")
        print("  C      → Set custom query")
        print("  X      → Quit")
        print("═" * 60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Auto-analyze in test mode
            if self.test_mode:
                self.test_counter += 1
                if self.test_counter >= 5:
                    print(f"[test] Auto-analyzing (frame #{self.test_counter})...")
                    self.analyze_frame_simple(frame)
                    self.test_counter = 0
            
            display = self.draw_on_frame(frame)
            cv2.imshow("Gemini Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('x'):
                break
            elif key == ord(' '):  # SPACE: analyze
                self.analyze_frame_simple(frame)
            elif key == ord('q'):  # Q: quick query
                self.query_frame(frame, "What is the main product in this image? List visible text. Give a brief description about the product.")
            elif key == ord('t'):  # T: toggle test mode
                self.test_mode = not self.test_mode
                state = "ON" if self.test_mode else "OFF"
                print(f"[test] Test mode {state}\n")
            elif key == ord('c'):  # C: custom query
                print("\n[test] Enter your query (press Enter to confirm):")
                query = input(">>> ").strip()
                if query:
                    self.custom_query = query
                    self.query_frame(frame, query)
                    self.custom_query = ""
        
        cap.release()
        cv2.destroyAllWindows()
        print("Gemini Test stopped.")


if __name__ == "__main__":
    test = GeminiTest()
    test.main()
