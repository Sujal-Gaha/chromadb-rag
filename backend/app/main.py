from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from app.model import QueryRequest
from evaluation.v3.run_evaluator import run_evaluation

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


@app.get("/api/documents")
async def list_documents(limit: int = 100):
    try:
        docs = rag_pipeline.document_store.filter_documents()[:limit]

        return [
            {
                "doc_id": doc.id,
                "filename": doc.meta.get("filename"),
                "score": doc.score if hasattr(doc, "score") else None,
                "content": doc.content,
            }
            for doc in docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluate-report", response_class=HTMLResponse)
async def evaluate_and_get_report(batch_size: int = Body(3, embed=True)):
    try:
        output_dir = "evaluation_results/v3"
        batch_result, summary, html_path = await run_evaluation(
            batch_size=batch_size,
            output_dir=output_dir,
            create_visualizations=True,
            create_reports=True,
            save_results=True,
        )
        if not html_path:
            raise HTTPException(
                status_code=500, detail="Failed to generate HTML report"
            )

        # Read the HTML content
        with open(html_path, "r") as f:
            html_content = f.read()

        # Return as HTML response
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate-summary", response_class=JSONResponse)
async def evaluate_and_get_summary(batch_size: int = Body(3, embed=True)):
    try:
        output_dir = "evaluation_results/v3"
        batch_result, summary, html_path = await run_evaluation(
            batch_size=batch_size,
            output_dir=output_dir,
            create_visualizations=False,
            create_reports=False,
            save_results=False,
        )
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/previous-report/{batch_id}", response_class=HTMLResponse)
async def get_previous_report(batch_id: str):
    try:
        output_dir = "evaluation_results/v3"  # Match your default
        report_path = f"{output_dir}/reports/report_{batch_id}.html"
        if not Path(report_path).exists():
            raise HTTPException(status_code=404, detail="Report not found")

        with open(report_path, "r") as f:
            html_content = f.read()

        return HTMLResponse(content=html_content)
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
    uvicorn.run(app, host="0.0.0.0", port=4321)
