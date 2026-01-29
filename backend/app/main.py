from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from .rag_pipeline import RAGPipeline

app = FastAPI()
rag_pipeline = RAGPipeline()


@app.on_event("startup")
async def startup():
    rag_pipeline.initialize()


@app.get("/")
async def home():
    return {"message": "Hello from the server"}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        result = await rag_pipeline.index_files(files)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(question: str):
    try:
        result = await rag_pipeline.query(question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/count")
async def document_count():
    count = await rag_pipeline.get_document_count()
    return {"count": count}


@app.delete("/api/documents")
async def clear_documents():
    result = await rag_pipeline.clear_documents()
    return result
