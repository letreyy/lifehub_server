import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ocr
import ai

app = FastAPI(title="LifeHub Server", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "LifeHub Server is running"}

@app.post("/api/ocr/receipt")
async def ocr_receipt(file: UploadFile = File(...)):
    print(f"\n--- [ocr_receipt] Received request. Filename: {file.filename}, Type: {file.content_type} ---")
    if not file.content_type.startswith("image/"):
        print("[ocr_receipt] Error: Unsupported content type.")
        raise HTTPException(status_code=400, detail="Only image files are supported for receipts.")
        
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        print("[ocr_receipt] Running PaddleOCR...")
        ocr_lines = ocr.extract_text_from_image(tmp_path)
        print(f"[ocr_receipt] PaddleOCR completed. Extracted {len(ocr_lines)} text lines.")
        
        print(f"[ocr_receipt] Sending text to Ollama (URL: {ai.OLLAMA_URL}, Model: {ai.MODEL_NAME})...")
        receipt_data = ai.process_receipt_ocr(ocr_lines)
        print(f"[ocr_receipt] AI processing completed. Result: {receipt_data}")
        return receipt_data
    except Exception as e:
        print(f"[ocr_receipt] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print("[ocr_receipt] Temporary file cleaned up.")

@app.post("/api/ocr/lab-results")
async def ocr_lab_results(file: UploadFile = File(...)):
    print(f"\n--- [ocr_lab_results] Received request. Filename: {file.filename}, Type: {file.content_type} ---")
    is_image = file.content_type.startswith("image/")
    is_pdf = file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf")
    
    if not (is_image or is_pdf):
        print("[ocr_lab_results] Error: Unsupported content type.")
        raise HTTPException(status_code=400, detail="Only image or PDF files are supported.")
        
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        print("[ocr_lab_results] Running PaddleOCR...")
        ocr_lines = ocr.extract_text_from_image(tmp_path)
        print(f"[ocr_lab_results] PaddleOCR completed. Extracted {len(ocr_lines)} text lines.")
        
        print(f"[ocr_lab_results] Sending text to Ollama (URL: {ai.OLLAMA_URL}, Model: {ai.MODEL_NAME})...")
        metrics_data = ai.process_lab_results_ocr(ocr_lines)
        print(f"[ocr_lab_results] AI processing completed. Result: {metrics_data}")
        return metrics_data
    except Exception as e:
        print(f"[ocr_lab_results] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print("[ocr_lab_results] Temporary file cleaned up.")

class UserDataModel(BaseModel):
    data: dict

@app.post("/api/ai/analyze-status")
async def analyze_status(payload: UserDataModel):
    try:
        recommendations = ai.generate_recommendations(payload.data)
        return {"analysis": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
