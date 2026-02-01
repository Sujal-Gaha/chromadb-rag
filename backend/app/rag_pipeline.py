import os
import time
import tempfile
import asyncio
import uuid

from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import UploadFile

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types.policy import DuplicatePolicy

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)

from .config import get_config
from .logger import get_logger

log = get_logger(__name__)


class RAGPipeline:
    def __init__(self):
        self.initialized = False
        self._lock = asyncio.Lock()

    def initialize(self):
        start_time = time.time()
        log.info("Initializing RAG Pipeline...")

        self.config = get_config()
        self.config.log_summary(log)

        os.makedirs(self.config.chromadb.persist_path, exist_ok=True)

        self.document_store = ChromaDocumentStore(
            collection_name=self.config.chromadb.collection_name,
            persist_path=self.config.chromadb.persist_path,
        )

        self._setup_indexing_pipeline()
        self._setup_query_pipeline()

        self.initialized = True
        log.info(f"Pipeline initialized in {time.time() - start_time:.2f}s")

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    def _build_indexing_pipeline(self):
        pipeline = Pipeline()

        pipeline.add_component(
            "splitter",
            DocumentSplitter(
                split_by="word",
                split_length=self.config.pipeline.chunk_size,
                split_overlap=self.config.pipeline.chunk_overlap,
            ),
        )
        pipeline.add_component(
            "embedder",
            OllamaDocumentEmbedder(
                model=self.config.ollama.embedding_model,
                url=self.config.ollama.server_url,
                timeout=self.config.ollama.timeout,
            ),
        )
        pipeline.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )

        pipeline.connect("splitter.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "writer.documents")

        return pipeline

    def _setup_indexing_pipeline(self):
        self.indexing_pipeline = self._build_indexing_pipeline()
        log.info("Indexing pipeline initialized")

    async def index_files(self, files: list[UploadFile]) -> dict[str, Any]:
        if not files:
            raise ValueError("No files provided")

        processed = []
        errors = []
        total_chunks = 0

        async with self._lock:
            for file in files:
                try:
                    filename = file.filename
                    suffix = Path(filename or "").suffix.lower()
                    source_id = str(uuid.uuid4())

                    content = await file.read()

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name

                    # Convert → Document WITH METADATA
                    if suffix == ".pdf":
                        docs = PyPDFToDocument().run(sources=[tmp_path])["documents"]
                    elif suffix == ".txt":
                        docs = TextFileToDocument().run(sources=[tmp_path])["documents"]
                    else:
                        raise ValueError(f"Unsupported file type: {suffix}")

                    # INJECT IDENTITY *BEFORE SPLITTING*
                    for doc in docs:
                        doc.meta.update(
                            {
                                "source_id": source_id,
                                "original_filename": filename,
                                "filename": filename,
                            }
                        )

                    result = self.indexing_pipeline.run(
                        {"splitter": {"documents": docs}}
                    )

                    written = result["writer"]["documents_written"]
                    total_chunks += written
                    processed.append(filename)

                    os.unlink(tmp_path)

                    log.info(f"Indexed {filename} → {written} chunks")

                except Exception as e:
                    log.error(f"Failed indexing {file.filename}: {e}")
                    errors.append({"filename": file.filename, "error": str(e)})

        return {
            "status": "success" if processed else "failed",
            "files_processed": len(processed),
            "chunks_created": total_chunks,
            "filenames": processed,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def _setup_query_pipeline(self):
        self.query_pipeline = Pipeline()

        retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.config.pipeline.top_k,
        )

        prompt = PromptBuilder(
            template="""
You are a helpful assistant that answers ONLY using the context.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ question }}

Answer:
""",
            required_variables=["documents", "question"],
        )

        llm = OllamaGenerator(
            model=self.config.ollama.model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

        embedder = OllamaTextEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

        self.query_pipeline.add_component("embedder", embedder)
        self.query_pipeline.add_component("retriever", retriever)
        self.query_pipeline.add_component("prompt", prompt)
        self.query_pipeline.add_component("llm", llm)

        self.query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.query_pipeline.connect("retriever.documents", "prompt.documents")
        self.query_pipeline.connect("prompt", "llm")

        log.info("Query pipeline initialized")

    async def query(self, question: str) -> dict[str, Any]:
        result = self.query_pipeline.run(
            {
                "embedder": {"text": question},
                "prompt": {"question": question},
            }
        )

        docs = result["retriever"]["documents"]

        sources = [
            {
                "filename": d.meta.get("original_filename", "unknown.txt"),
                "source_id": d.meta.get("source_id"),
                "score": getattr(d, "score", 0.0),
            }
            for d in docs
        ]

        return {
            "reply": result["llm"]["replies"][0],
            "sources": sources,
            "retrieved_documents": len(docs),
        }

    # ------------------------------------------------------------------

    async def clear_documents(self):
        async with self._lock:
            docs = self.document_store.filter_documents()
            if docs:
                self.document_store.delete_documents([d.id for d in docs])
        return {"status": "cleared"}

    @asynccontextmanager
    async def batch_indexing(self):
        async with self._lock:
            yield self
