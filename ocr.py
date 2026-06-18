import os
import cv2
from paddleocr import PaddleOCR

# Initialize OCR engine lazily
_ocr = None

def detect_and_decode_qr(image_path: str) -> str:
    """
    Attempts to detect and decode a QR code in the image using pyzbar and PIL,
    falling back to OpenCV QRCodeDetector if pyzbar fails.
    """
    try:
        if not os.path.exists(image_path):
            return ""
            
        # 1. Try pyzbar (highly accurate)
        from pyzbar.pyzbar import decode
        from PIL import Image
        
        print("[QR] Attempting pyzbar QR decode...")
        with Image.open(image_path) as img:
            decoded_objects = decode(img)
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    data = obj.data.decode('utf-8')
                    print(f"[QR] Detected QR code using pyzbar: {data}")
                    return data
                    
        # 2. Try OpenCV (fallback)
        print("[QR] pyzbar found no QR codes. Trying OpenCV fallback...")
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        detector = cv2.QRCodeDetector()
        
        # Try direct decoding
        val, points, straight_qrcode = detector.detectAndDecode(img)
        if val:
            print(f"[QR] Detected QR code using OpenCV fallback (direct): {val}")
            return val
            
        # Try grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, points, straight_qrcode = detector.detectAndDecode(gray)
        if val:
            print(f"[QR] Detected QR code using OpenCV fallback (grayscale): {val}")
            return val
            
        # Try resizing
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            val, points, straight_qrcode = detector.detectAndDecode(resized)
            if val:
                print(f"[QR] Detected QR code using OpenCV fallback (resized): {val}")
                return val
    except Exception as e:
        print(f"[QR] Error decoding QR code: {e}")
    return ""

def get_ocr_engine():
    global _ocr
    if _ocr is None:
        # lang='ru' enables Russian model, use_angle_cls=True corrects rotation
        _ocr = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False)
    return _ocr

def extract_text_from_image(image_path: str):
    """
    Extracts text blocks from an image and returns them sorted by reading layout.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    engine = get_ocr_engine()
    result = engine.ocr(image_path, cls=True)
    
    lines = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text, confidence = line[1]
            
            x_center = sum([p[0] for p in box]) / 4.0
            y_center = sum([p[1] for p in box]) / 4.0
            
            lines.append({
                "text": text,
                "confidence": float(confidence),
                "x_center": x_center,
                "y_center": y_center
            })
            
    # Sort lines: group items that are vertically close (on the same row)
    # using a tolerance of 15 pixels, then sort left-to-right.
    lines_sorted = sorted(lines, key=lambda l: (l["y_center"] // 15, l["x_center"]))
    return lines_sorted
