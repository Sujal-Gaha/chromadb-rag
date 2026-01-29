from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

import uvicorn

from app.model import QueryRequest

from .rag_pipeline import RAGPipeline

app = FastAPI()
rag_pipeline = RAGPipeline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    rag_pipeline.initialize()


@app.get("/")
async def home():
    return {"message": "Hello from the server"}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accept multiple files from form-data
    Example frontend usage:
    const formData = new FormData();
    formData.append("files", file1);
    formData.append("files", file2);

    fetch("/api/upload", {
        method: "POST",
        body: formData
    })
    """
    try:
        result = await rag_pipeline.index_files(files)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/single")
async def upload_single_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = None,
):
    """
    Alternative endpoint for single file upload
    """
    try:
        result = await rag_pipeline.index_files([file])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(payload: QueryRequest):
    try:
        result = await rag_pipeline.query(payload.question)
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


@app.get("/api/files")
async def list_uploaded_files():
    try:
        docs = rag_pipeline.document_store.filter_documents()

        files = {}

        for doc in docs:
            filename = doc.meta.get("filename", "unknown")
            content = doc.content or ""

            if filename not in files:
                files[filename] = {
                    "filename": filename,
                    "file_type": doc.meta.get("file_type"),
                    "upload_time": doc.meta.get("upload_time"),
                    "chunks": [],
                }

            files[filename]["chunks"].append(
                {
                    "doc_id": doc.id,
                    "content_preview": (
                        content[:120] + "..." if len(content) > 120 else content
                    ),
                }
            )

        return {"total_files": len(files), "files": list(files.values())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": rag_pipeline.initialized,
        "document_count": await rag_pipeline.get_document_count(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
