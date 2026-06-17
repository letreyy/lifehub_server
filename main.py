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
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported for receipts.")
        
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        ocr_lines = ocr.extract_text_from_image(tmp_path)
        receipt_data = ai.process_receipt_ocr(ocr_lines)
        return receipt_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/ocr/lab-results")
async def ocr_lab_results(file: UploadFile = File(...)):
    # Check if PDF or image
    is_image = file.content_type.startswith("image/")
    is_pdf = file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf")
    
    if not (is_image or is_pdf):
        raise HTTPException(status_code=400, detail="Only image or PDF files are supported.")
        
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        # PaddleOCR natively supports PDF file path as input (lang='ru' page_num is handled)
        ocr_lines = ocr.extract_text_from_image(tmp_path)
        metrics_data = ai.process_lab_results_ocr(ocr_lines)
        return metrics_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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
