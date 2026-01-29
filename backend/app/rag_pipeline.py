import os
import time
import tempfile
import asyncio

from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import UploadFile

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner

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

            self._initialize_embedders()

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

    def _initialize_embedders(self):
        log.info("Initializing Ollama embedders")
        log.info(f"  Model: {self.config.ollama.embedding_model}")
        log.info(f"  Server: {self.config.ollama.server_url}")

        self.doc_embedder = OllamaDocumentEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

        self.text_embedder = OllamaTextEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

    def _setup_indexing_pipeline(self):
        """Setup indexing pipeline for processing documents"""
        self.indexing_pipeline = Pipeline()

        pdf_converter = PyPDFToDocument()
        txt_converter = TextFileToDocument()

        joiner = DocumentJoiner()

        splitter = DocumentSplitter(
            split_by="sentence",
            split_length=self.config.pipeline.chunk_size,
            split_overlap=self.config.pipeline.chunk_overlap,
        )

        writer = DocumentWriter(document_store=self.document_store)

        embedder = OllamaDocumentEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

        self.indexing_pipeline.add_component("pdf_converter", pdf_converter)
        self.indexing_pipeline.add_component("txt_converter", txt_converter)
        self.indexing_pipeline.add_component("joiner", joiner)
        self.indexing_pipeline.add_component("splitter", splitter)
        self.indexing_pipeline.add_component("embedder", embedder)
        self.indexing_pipeline.add_component("writer", writer)

        self.indexing_pipeline.connect("pdf_converter.documents", "joiner.documents")
        self.indexing_pipeline.connect("txt_converter.documents", "joiner.documents")
        self.indexing_pipeline.connect("joiner.documents", "splitter.documents")
        self.indexing_pipeline.connect("splitter.documents", "embedder.documents")
        self.indexing_pipeline.connect("embedder.documents", "writer.documents")

        log.info("Indexing pipeline initialized")

    def _setup_query_pipeline(self):
        """Setup query pipeline for answering questions"""
        self.query_pipeline = Pipeline()

        retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store, top_k=self.config.pipeline.top_k
        )

        template = """
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}
        
        Question: {{ question }}
        
        Please provide a clear and concise answer based on the context above. 
        If the context doesn't contain enough information to answer the question, say so.
        
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

    async def _process_uploaded_file(self, file: UploadFile) -> list[Document]:
        """Process a single uploaded file asynchronously"""
        if not file.filename:
            raise ValueError("File must have a filename")

        log.info(f"Processing file: {file.filename}")
        suffix = Path(file.filename).suffix.lower()

        content = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode="wb"
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                result = self.indexing_pipeline.run(
                    {"pdf_converter": {"sources": [tmp_path]}}
                )
                documents = result["pdf_converter"]["documents"]
            elif suffix == ".txt":
                result = self.indexing_pipeline.run(
                    {"txt_converter": {"sources": [tmp_path]}}
                )
                documents = result["txt_converter"]["documents"]
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix}. Supported types: .pdf, .txt"
                )

            for doc in documents:
                doc.meta.update(
                    {
                        "filename": file.filename,
                        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "file_type": suffix,
                    }
                )

            log.info(f"Extracted {len(documents)} document(s) from {file.filename}")
            return documents

        except Exception as e:
            log.error(f"Error processing {file.filename}: {str(e)}")
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except Exception as e:
                log.warning(f"Could not delete temp file {tmp_path}: {e}")

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
                try:
                    log.info(f"[{idx}/{len(files)}] Processing: {file.filename}")

                    documents = await self._process_uploaded_file(file)

                    if documents:
                        split_result = self.indexing_pipeline.run(
                            {"splitter": {"documents": documents}}
                        )
                        split_docs = split_result["splitter"]["documents"]

                        embed_result = self.indexing_pipeline.run(
                            {"embedder": {"documents": split_docs}}
                        )
                        embedded_docs = embed_result["embedder"]["documents"]

                        self.indexing_pipeline.run(
                            {"writer": {"documents": embedded_docs}}
                        )

                        total_chunks_created += len(embedded_docs)
                        log.info(f"  Created {len(embedded_docs)} chunks")

                    processed_files.append(file.filename)

                except Exception as e:
                    log.error(
                        f"[{idx}/{len(files)}] Failed: {file.filename} - {str(e)}"
                    )
                    errors.append({"filename": file.filename, "error": str(e)})

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
        """Query the indexed documents"""
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        log.info(f"Querying: '{question[:100]}...'")
        start_time = time.time()

        try:
            result = self.query_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                }
            )

            reply = result["llm"]["replies"][0]
            retrieved_docs = result.get("retriever", {}).get("documents", [])

            sources = []
            for doc in retrieved_docs[:3]:
                sources.append(
                    {
                        "filename": doc.meta.get("filename", "unknown"),
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

            return {
                "reply": reply,
                "question": question,
                "retrieved_documents": len(retrieved_docs),
                "sources": sources,
                "elapsed_time": f"{elapsed_time:.2f}s",
            }

        except Exception as e:
            log.error(f"Query failed: {str(e)}")
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
