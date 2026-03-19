
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

from app.core.ocr_processor import OCRProcessor
from app.models.model_loader import ModelLoader
from app.core.vector_store import VectorStore

app = FastAPI(title="Document Intelligence API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ocr_processor = OCRProcessor()
models = ModelLoader()
vector_store = VectorStore()

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class ClassificationRequest(BaseModel):
    text: str
    labels: List[str]

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract text
        if file.filename.endswith('.pdf'):
            text = ocr_processor.process_pdf(tmp_path)
        else:
            text = ocr_processor.process_image(tmp_path)
        
        # Generate summary
        summary = models.summarize_text(text[:2000])  # Limit for demo
        
        # Extract entities
        entities = models.extract_entities(text[:1000])
        
        # Store in vector DB
        doc_id = vector_store.add_document(text, {
            "filename": file.filename,
            "summary": summary
        })
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "summary": summary,
            "entities": entities,
            "text_preview": text[:500] + "..."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(request: QueryRequest):
    """Search through processed documents"""
    results = vector_store.search_similar(
        request.query, 
        n_results=request.n_results
    )
    return {"results": results}

@app.post("/classify")
async def classify_text(request: ClassificationRequest):
    """Classify text into categories"""
    result = models.classify_document(request.text, request.labels)
    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}