import os
import time
import tempfile
import asyncio
import re
from haystack.utils.device import ComponentDevice
import torch

from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import UploadFile

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import SentenceTransformersSimilarityRanker

from haystack.document_stores.types.policy import DuplicatePolicy
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)

from utils.logger import get_logger
from utils.config import get_config

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
            self.config.log_summary(logger=log)

            os.makedirs(name=self.config.chromadb.persist_path, exist_ok=True)
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

        self.pdf_indexing_pipeline = build_pipeline(converter=PyPDFToDocument())
        self.txt_indexing_pipeline = build_pipeline(converter=TextFileToDocument())

        log.info("Indexing pipelines (PDF, TXT) initialized")

    def _setup_query_pipeline(self):
        self.query_pipeline = Pipeline()

        retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.config.pipeline.top_k,
        )

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = ComponentDevice.from_str(device_str=device_str)
        reranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5, device=device
        )
        log.info(f"Reranker initialized on device: {device}")

        template = """
        You are a strictly faithful assistant. Follow these rules:
        
        1. Base your answer **only** on the provided context — do not add external knowledge or assumptions.
        2. Use wording and facts as close as possible to the context.
        3. If the context has partial information, answer with what is available and note any gaps (e.g., "The context mentions..., but does not fully explain...").
        4. If **no relevant information** exists in the context, say:  
           "The provided context does not contain information to answer this question."
        5. Keep answers concise and factual.
        
        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}
        
        Question: {{ question }}
        
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
                "repeat_penalty": 1.1,
            },
        )

        text_embedder = OllamaTextEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
        )

        # Add components
        self.query_pipeline.add_component(name="text_embedder", instance=text_embedder)
        self.query_pipeline.add_component(name="retriever", instance=retriever)
        self.query_pipeline.add_component(name="reranker", instance=reranker)
        self.query_pipeline.add_component(
            name="prompt_builder", instance=prompt_builder
        )
        self.query_pipeline.add_component(name="llm", instance=ollama_generator)

        # Connect components
        self.query_pipeline.connect(
            sender="text_embedder.embedding", receiver="retriever.query_embedding"
        )
        self.query_pipeline.connect(
            sender="retriever.documents", receiver="reranker.documents"
        )
        self.query_pipeline.connect(
            sender="reranker.documents", receiver="prompt_builder.documents"
        )
        self.query_pipeline.connect(sender="prompt_builder", receiver="llm")

        log.info("Query pipeline initialized")

    async def index_files(self, files: list[UploadFile]) -> dict[str, Any]:
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
                    # Capture original filename FIRST
                    original_filename = file.filename or f"unknown-{idx}.txt"
                    log.info(f"[{idx}/{len(files)}] Processing: {original_filename}")

                    suffix = Path(file.filename or "").suffix.lower()
                    content = await file.read()

                    # Create temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, mode="wb"
                    ) as tmp_file:
                        tmp_file.write(content)
                        tmp_path = tmp_file.name

                    log.debug(f"Created temp file: {tmp_path}")

                    meta = {
                        "original_filename": original_filename,
                        "filename": original_filename,
                    }

                    # Run indexing pipeline
                    if suffix == ".pdf":
                        result = self.pdf_indexing_pipeline.run(
                            data={"converter": {"sources": [tmp_path], "meta": meta}}
                        )
                    elif suffix == ".txt":
                        result = self.txt_indexing_pipeline.run(
                            data={"converter": {"sources": [tmp_path], "meta": meta}}
                        )
                    else:
                        raise ValueError(
                            f"Unsupported file type: {suffix}. Supported types: .pdf, .txt"
                        )

                    written = result["writer"].get("documents_written", 0)
                    total_chunks_created += written
                    log.info(f"Created {written} chunks from {original_filename}")

                    processed_files.append(original_filename)

                except Exception as e:
                    log.error(
                        f"[{idx}/{len(files)}] ✗ Failed: {file.filename} - {str(e)}"
                    )
                    errors.append({"filename": file.filename, "error": str(e)})

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(path=tmp_path)
                            log.debug(f"Deleted temp file: {tmp_path}")
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
                    "reranker": {"query": question},
                },
                include_outputs_from=set(["retriever", "llm", "text_embedder"]),
            )

            reply = result["llm"]["replies"][0]
            if not reply.strip():
                log.warning(
                    f"Empty generation for question: {question}. Raw meta: {result['llm'].get('meta', 'No meta')}"
                )
                reply = "The provided context does not contain sufficient information to answer this question."

            retrieved_docs = result.get("retriever", {}).get("documents", [])

            # Build sources with proper filename extraction
            sources = []
            if retrieved_docs:
                for doc in retrieved_docs:
                    filename = self._extract_filename_from_metadata(meta=doc.meta)

                    # If metadata fails or returns temp/unknown, extract from content
                    if (
                        filename.startswith("tmp")
                        or filename == "unknown.txt"
                        or filename == "unknown"
                    ):
                        content_filename = self._extract_original_filename_from_content(
                            content=doc.content
                        )
                        if content_filename != "unknown-document.txt":
                            filename = content_filename

                    sources.append(
                        {
                            "filename": filename,
                            "content": (
                                doc.content[:200] + "..."
                                if len(doc.content) > 200
                                else doc.content
                            ),
                            "score": doc.score if hasattr(doc, "score") else 0.0,
                            "original_meta": doc.meta,
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
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        try:
            all_docs = self.document_store.filter_documents()
            return len(all_docs)
        except Exception as e:
            log.error(f"Failed to get document count: {str(e)}")
            return 0

    async def clear_documents(self) -> dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        async with self._lock:
            try:
                all_docs = self.document_store.filter_documents()

                if all_docs:
                    doc_ids = [doc.id for doc in all_docs]
                    self.document_store.delete_documents(document_ids=doc_ids)
                    log.info(f"Deleted {len(all_docs)} documents")
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

    def _extract_original_filename_from_content(self, content: str) -> str:
        if not content:
            return "unknown-document.txt"

        try:
            lines = content.split("\n")

            # Look for **Title: ...** format in first 10 lines
            for line in lines[:10]:
                # Pattern 1: **Title: Some Title**
                if "**Title:" in line:
                    title_match = re.search(r"\*\*Title: (.+?)\*\*", line)
                    if title_match:
                        title = title_match.group(1).strip()
                        return self._convert_title_to_filename(title)

                # Pattern 2: Title: Some Title (without markdown)
                elif "Title:" in line and "**" not in line:
                    title = line.replace("Title:", "").strip()
                    if title:
                        return self._convert_title_to_filename(title)

            # If no title found, this is likely a continuation chunk
            # Use content hash for uniqueness
            log.debug(f"Could not extract title from content: {content[:100]}...")
            return f"document-{abs(hash(content[:100]))}.txt"

        except Exception as e:
            log.warning(f"Failed to extract filename from content: {e}")
            return "unknown-document.txt"

    def _convert_title_to_filename(self, title: str) -> str:
        filename = title.lower()
        filename = filename.replace(" ", "-")
        filename = re.sub(r"[^\w\-\.]", "", filename)

        if not filename.endswith(".txt"):
            filename += ".txt"

        return filename

    def _extract_filename_from_metadata(self, meta: dict) -> str:
        for field in ["original_filename", "filename", "file_path", "source"]:
            if field in meta and meta[field]:
                file_path = meta[field]
                filename = os.path.basename(str(file_path))

                # Skip temp files
                if filename.startswith("tmp") and len(filename) < 20:
                    continue

                return filename

        return "unknown.txt"

    @asynccontextmanager
    async def batch_indexing(self):
        async with self._lock:
            yield self
