import os
import time
import tempfile
import asyncio

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

        try:
            self.config = get_config()
            self.config.log_summary(log)

            os.makedirs(self.config.chromadb.persist_path, exist_ok=True)
            log.info(f"ChromaDB directory ensured: {self.config.chromadb.persist_path}")

            self._setup_chromadb()

            self._setup_indexing_pipeline()
            self._setup_query_pipeline()

            elapsed = time.time() - start_time

            log.info("=" * 70)
            log.info(f"Pipeline setup completed in {elapsed:.2f}s")
            log.info("=" * 70)

            self.initialized = True

        except Exception as e:
            log.error(f"Failed to initialize pipeline: {str(e)}")
            raise

    def _setup_chromadb(self):
        log.info(f"Setting up ChromaDB at: {self.config.chromadb.persist_path}")

        try:
            self.document_store = ChromaDocumentStore(
                collection_name=self.config.chromadb.collection_name,
                persist_path=self.config.chromadb.persist_path,
            )

            test_count = len(self.document_store.filter_documents())
            log.info(f"ChromaDB initialized. Existing documents: {test_count}")

        except Exception as e:
            log.error(f"ChromaDB setup failed: {str(e)}")
            self.document_store = ChromaDocumentStore(
                collection_name=self.config.chromadb.collection_name,
            )
            log.warning("ChromaDB initialized without persistence (in-memory)")

    def _setup_indexing_pipeline(self):
        """Setup indexing pipelines for PDF and TXT files"""

        def build_pipeline(converter):
            pipeline = Pipeline()

            pipeline.add_component("converter", converter)
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
                    document_store=self.document_store, policy=DuplicatePolicy.OVERWRITE
                ),
            )

            pipeline.connect("converter.documents", "splitter.documents")
            pipeline.connect("splitter.documents", "embedder.documents")
            pipeline.connect("embedder.documents", "writer.documents")

            return pipeline

        self.pdf_indexing_pipeline = build_pipeline(PyPDFToDocument())
        self.txt_indexing_pipeline = build_pipeline(TextFileToDocument())

        log.info("Indexing pipelines (PDF, TXT) initialized")

    def _setup_query_pipeline(self):
        """Setup query pipeline for answering questions"""
        self.query_pipeline = Pipeline()

        retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store, top_k=self.config.pipeline.top_k
        )

        template = """
        You are a helpful assistant that answers questions based ONLY on the provided context.
        
        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}
        
        Question: {{ question }}

        Instructions:
        1. Answer ONLY using information from the context above.
        2. If the context doesn't contain relevant information, say "I cannot answer based on the provided documents."
        3. Be precise and concise.
        4. Include relevant details from the context.
        
        Answer:
        """

        prompt_builder = PromptBuilder(
            template=template, required_variables=["documents", "question"]
        )

        ollama_generator = OllamaGenerator(
            model=self.config.ollama.model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
            generation_kwargs={
                "num_predict": self.config.llm.num_predict,
                "temperature": self.config.llm.temperature,
                "num_ctx": self.config.llm.num_ctx,
                "top_p": self.config.llm.top_p,
            },
        )

        text_embedder = OllamaTextEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

        self.query_pipeline.add_component("text_embedder", text_embedder)
        self.query_pipeline.add_component("retriever", retriever)
        self.query_pipeline.add_component("prompt_builder", prompt_builder)
        self.query_pipeline.add_component("llm", ollama_generator)

        self.query_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )
        self.query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "llm")

        log.info("Query pipeline initialized")

    async def index_files(self, files: list[UploadFile]) -> dict[str, Any]:
        """Index uploaded files asynchronously"""
        if not files:
            raise ValueError("No files provided for indexing")

        log.info("=" * 70)
        log.info(f"Starting indexing for {len(files)} file(s)")
        log.info("=" * 70)
        start = time.time()

        processed_files = []
        errors = []
        total_chunks_created = 0

        async with self._lock:
            for idx, file in enumerate(files, 1):
                tmp_path = None
                try:
                    log.info(f"[{idx}/{len(files)}] Processing: {file.filename}")

                    suffix = Path(file.filename or "").suffix.lower()
                    content = await file.read()

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, mode="wb"
                    ) as tmp_file:
                        tmp_file.write(content)
                        tmp_path = tmp_file.name

                    if suffix == ".pdf":
                        result = self.pdf_indexing_pipeline.run(
                            {"converter": {"sources": [tmp_path]}}
                        )
                    elif suffix == ".txt":
                        result = self.txt_indexing_pipeline.run(
                            {"converter": {"sources": [tmp_path]}}
                        )
                    else:
                        raise ValueError(
                            f"Unsupported file type: {suffix}. Supported types: .pdf, .txt"
                        )

                    written = result["writer"].get("documents_written", 0)
                    total_chunks_created += written

                    log.info(f"  Created {written} chunks")
                    processed_files.append(file.filename)

                except Exception as e:
                    log.error(
                        f"[{idx}/{len(files)}] Failed: {file.filename} - {str(e)}"
                    )
                    errors.append({"filename": file.filename, "error": str(e)})
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception as e:
                            log.warning(f"Could not delete temp file {tmp_path}: {e}")

        elapsed = time.time() - start

        log.info("=" * 70)
        log.info(f"Indexing completed in {elapsed:.2f}s")
        log.info(f"Files processed: {len(processed_files)}/{len(files)}")
        log.info(f"Total chunks created: {total_chunks_created}")
        log.info(f"Errors: {len(errors)}")
        log.info("=" * 70)

        return {
            "status": "success" if processed_files else "failed",
            "files_processed": len(processed_files),
            "total_files": len(files),
            "filenames": processed_files,
            "chunks_created": total_chunks_created,
            "embedding_provider": "ollama",
            "embedding_model": self.config.ollama.embedding_model,
            "errors": errors,
            "elapsed_time": f"{elapsed:.2f}s",
        }

    async def query(self, question: str) -> dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        log.info(f"Querying: '{question[:100]}...'")
        start_time = time.time()

        try:
            result = self.query_pipeline.run(
                data={
                    "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                },
                include_outputs_from=set(["retriever", "llm", "text_embedder"]),
            )

            log.info("=" * 80)
            log.info("QUERY RESULT STRUCTURE:")
            log.info(f"Result keys: {list(result.keys())}")

            for key in result.keys():
                log.info(f"  {key}: {type(result[key])}")
                if isinstance(result[key], dict):
                    log.info(f"    Sub-keys: {list(result[key].keys())}")

            # Check retriever specifically
            if "retriever" in result:
                log.info(f"Retriever output: {result['retriever']}")
                if "documents" in result["retriever"]:
                    docs = result["retriever"]["documents"]
                    log.info(f"Number of documents from retriever: {len(docs)}")
                    if docs:
                        log.info(f"First document type: {type(docs[0])}")
                        log.info(f"First document: {docs[0]}")

            log.info("=" * 80)

            reply = result["llm"]["replies"][0]
            retrieved_docs = result.get("retriever", {}).get("documents", [])

            sources = []
            for doc in retrieved_docs[:3]:
                # Extract filename from meta
                filename = (
                    doc.meta.get("file_path")
                    or doc.meta.get("filename")
                    or doc.meta.get("name")
                    or "unknown"
                )

                # Clean filename
                if "/" in filename:
                    filename = filename.split("/")[-1]

                log.info(f"Processing document with filename: {filename}")

                sources.append(
                    {
                        "filename": filename,
                        "score": doc.score if hasattr(doc, "score") else None,
                        "content": (
                            doc.content[:200] + "..."
                            if len(doc.content) > 200
                            else doc.content
                        ),
                    }
                )

            elapsed_time = time.time() - start_time
            log.info(f"Query completed in {elapsed_time:.2f}s")
            log.info(f"Retrieved {len(retrieved_docs)} documents")
            log.info(f"Created {len(sources)} sources")

            return {
                "reply": reply,
                "question": question,
                "retrieved_documents": len(retrieved_docs),
                "sources": sources,
                "elapsed_time": f"{elapsed_time:.2f}s",
            }

        except Exception as e:
            log.error(f"Query failed: {str(e)}", exc_info=True)
            raise

    async def get_document_count(self) -> int:
        """Get count of indexed documents"""
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        try:
            all_docs = self.document_store.filter_documents()
            return len(all_docs)
        except Exception as e:
            log.error(f"Failed to get document count: {str(e)}")
            return 0

    async def clear_documents(self) -> dict[str, Any]:
        """Clear all indexed documents"""
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        async with self._lock:
            try:
                all_docs = self.document_store.filter_documents()

                if all_docs:
                    doc_ids = [doc.id for doc in all_docs]
                    self.document_store.delete_documents(document_ids=doc_ids)
                    log.info("No documents to delete")
                else:
                    log.info("No documents to delete")

                self.document_store = ChromaDocumentStore(
                    collection_name=self.config.chromadb.collection_name,
                    persist_path=self.config.chromadb.persist_path,
                )

                self._setup_indexing_pipeline()
                self._setup_query_pipeline()

                log.info("All documents cleared successfully")

                return {
                    "status": "success",
                    "message": "All documents cleared",
                    "documents_deleted": len(all_docs) if all_docs else 0,
                }

            except Exception as e:
                log.error(f"Failed to clear documents: {str(e)}")
                return {"status": "error", "message": str(e)}

    @asynccontextmanager
    async def batch_indexing(self):
        """Context manager for batch indexing operations"""
        async with self._lock:
            yield self
