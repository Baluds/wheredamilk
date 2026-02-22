"""
vision/gemini.py — Google Gemini Vision API for product detection and description.

Uses Google's Gemini API to analyze video frames and identify products with detailed information.
More accurate than YOLO for product identification and label reading.

Setup:
    1. Get API key from: https://ai.google.dev/
    2. Add to .env: GEMINI_API_KEY=your_key_here
    3. Install: pip install google-generativeai pillow

Usage:
    analyzer = GeminiAnalyzer()
    results = analyzer.analyze_frame(frame)
    # Returns: {"products": [...], "text": "...", "description": "..."}
"""

import os
import base64
import io
from typing import Dict, Any

# ── Gemini setup ──────────────────────────────────────────────────────────────
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = bool(_GEMINI_KEY)
    if _GEMINI_AVAILABLE:
        genai.configure(api_key=_GEMINI_KEY)
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class GeminiAnalyzer:
    def __init__(self, model_name: str = "gemini-3-flash-preview", max_output_tokens: int = 500):
        """Initialize Gemini vision analyzer.
        
        Args:
            model_name: Gemini model to use (flash is fast/cheap, pro is more accurate)
            max_output_tokens: Maximum tokens in response (default 500 for speed)
        """
        self.model_name = model_name
        self.model = None
        self.max_output_tokens = max_output_tokens
        
        # Generation config for faster responses with minimal thinking
        self.generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.7,  # Balanced: not too random, not too deterministic
            "top_p": 0.95,
            "top_k": 40,
        }
        
        print("[gemini] ========== Gemini Initialization ==========")
        print(f"[gemini] Config: max_tokens={max_output_tokens}, temp=0.7")
        
        if not _GEMINI_AVAILABLE:
            print("[gemini] ❌ Gemini API key not found.")
            print("[gemini] Set GEMINI_API_KEY in .env or environment variables")
            return
        
        if not _PIL_AVAILABLE:
            print("[gemini] ❌ PIL not installed. Install: pip install pillow")
            return
        
        try:
            self.model = genai.GenerativeModel(
                model_name,
                generation_config=self.generation_config
            )
            print(f"[gemini] ✅ Gemini {model_name} ready")
        except Exception as e:
            print(f"[gemini] ❌ Failed to initialize Gemini: {e}")
        
        print("[gemini] ========== Initialization Complete ==========\n")
    
    def available(self) -> bool:
        """Check if Gemini is ready to use."""
        return self.model is not None
    
    def _frame_to_base64(self, frame) -> str:
        """Convert OpenCV BGR frame to base64 JPEG."""
        import cv2
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', rgb_frame)
        
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_frame(self, frame, detailed: bool = False) -> Dict[str, Any]:
        """Analyze a frame and identify products/objects.
        
        Args:
            frame: OpenCV BGR frame
            detailed: If True, request more detailed analysis (slower, costs more)
        
        Returns:
            {
                "success": bool,
                "products": [{"name": str, "description": str, "confidence": str}],
                "text": str,  # Any visible text
                "description": str,  # Overall scene description
                "error": str (if any)
            }
        """
        if not self.available():
            return {
                "success": False,
                "error": "Gemini not available",
                "products": [],
                "text": "",
                "description": ""
            }
        
        try:
            # Convert frame to base64
            frame_b64 = self._frame_to_base64(frame)
            
            # Build prompt
            if detailed:
                prompt = """Analyze this image and provide:
1. List all visible products/objects with descriptions
2. Extract any visible text or labels
3. General scene description
4. Give more information about each product (e.g., type, brand, use, weight,ingredients, etc.)

Format your response as:
PRODUCTS:
- [product name]: [description and confidence]
- [product name]: [description and confidence]

TEXT:
[any visible text or labels]

DESCRIPTION:
[overall scene]"""
            else:
                prompt = """What products or objects do you see in this image? 
List 3-5 main items with brief descriptions.
Also extract any visible text on labels or packaging.

Format:
PRODUCTS: [list with descriptions]
TEXT: [visible text if any]"""
            
            # Call Gemini API
            print("[gemini] 🔄 Analyzing frame...")
            response = self.model.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": frame_b64
                },
                prompt
            ], generation_config=self.generation_config)
            
            text = response.text
            print(f"[gemini] ✅ Analysis complete")
            
            # Parse response
            return self._parse_response(text)
        
        except Exception as e:
            print(f"[gemini] ❌ Error analyzing frame: {e}")
            return {
                "success": False,
                "error": str(e),
                "products": [],
                "text": "",
                "description": ""
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response into structured data."""
        result = {
            "success": True,
            "products": [],
            "text": "",
            "description": "",
            "raw": response_text
        }
        
        lines = response_text.split('\n')
        section = None
        products_lines = []
        text_lines = []
        desc_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("PRODUCTS:"):
                section = "products"
                products_lines.append(line.replace("PRODUCTS:", "").strip())
            elif line.startswith("TEXT:"):
                section = "text"
                text_lines.append(line.replace("TEXT:", "").strip())
            elif line.startswith("DESCRIPTION:"):
                section = "description"
                desc_lines.append(line.replace("DESCRIPTION:", "").strip())
            elif section == "products" and line:
                if line.startswith("-"):
                    products_lines.append(line)
            elif section == "text" and line:
                text_lines.append(line)
            elif section == "description" and line:
                desc_lines.append(line)
        
        # Parse products
        for prod_line in products_lines:
            if prod_line.startswith("-"):
                prod_line = prod_line[1:].strip()
            if ":" in prod_line:
                name, desc = prod_line.split(":", 1)
                result["products"].append({
                    "name": name.strip(),
                    "description": desc.strip(),
                    "confidence": "high"
                })
            elif prod_line:
                result["products"].append({
                    "name": prod_line,
                    "description": "",
                    "confidence": "medium"
                })
        
        result["text"] = " ".join(text_lines).strip()
        result["description"] = " ".join(desc_lines).strip()
        
        return result
    
    def identify_product(self, frame, query: str = "") -> Dict[str, Any]:
        """Quickly identify a specific product or answer a question about the frame.
        
        Args:
            frame: OpenCV BGR frame
            query: Optional specific question (e.g., "Is there any milk in this image?")
        
        Returns:
            Response with product details and any visible text
        """
        if not query:
            query = "What is the main product visible in this image?"
        
        if not self.available():
            return {
                "success": False,
                "error": "Gemini not available",
                "answer": ""
            }
        
        try:
            frame_b64 = self._frame_to_base64(frame)
            
            print(f"[gemini] 🔄 Processing query: {query}")
            response = self.model.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": frame_b64
                },
                query
            ], generation_config=self.generation_config)
            
            print(f"[gemini] ✅ Query complete")
            
            return {
                "success": True,
                "answer": response.text,
                "query": query
            }
        
        except Exception as e:
            print(f"[gemini] ❌ Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": ""
            }
