from haystack_integrations.document_stores.chroma import ChromaDocumentStore
import json

from utils.logger import get_logger

log = get_logger(__name__)


def diagnose_documents():
    log.info("=" * 80)
    log.info("CHROMADB DOCUMENT METADATA DIAGNOSTIC")
    log.info("=" * 80)

    document_store = ChromaDocumentStore(
        collection_name="document_collection",
        persist_path="./chroma_data",
    )

    # Get all documents
    all_docs = document_store.filter_documents()

    log.info(f"Total documents in database: {len(all_docs)}")

    # Group by filename
    by_filename = {}
    unknown_count = 0
    temp_count = 0

    for doc in all_docs:
        # Extract filename from metadata
        original_filename = doc.meta.get("original_filename")
        filename = doc.meta.get("filename")
        file_path = doc.meta.get("file_path")

        # Determine which filename to use
        if original_filename:
            key = original_filename
        elif filename:
            key = filename
        elif file_path:
            key = file_path
        else:
            key = "UNKNOWN"
            unknown_count += 1

        if key.startswith("tmp"):
            temp_count += 1

        if key not in by_filename:
            by_filename[key] = []

        by_filename[key].append(
            {
                "doc_id": doc.id[:16] + "...",
                "content_preview": doc.content[:80] + "..." if doc.content else "",
                "meta": doc.meta,
            }
        )

    log.info("FILENAME DISTRIBUTION:")
    log.info("-" * 80)
    for filename, docs in sorted(by_filename.items()):
        log.info(f"{filename}")
        log.info(f"Chunks: {len(docs)}")

        # Show first chunk's metadata
        if docs:
            first_doc = docs[0]
            log.info(f"First chunk ID: {first_doc['doc_id']}")
            log.info(f"Content: {first_doc['content_preview']}")
            log.info(f"Metadata keys: {list(first_doc['meta'].keys())}")

            # Check for original_filename
            if "original_filename" in first_doc["meta"]:
                log.info(
                    f"as original_filename: {first_doc['meta']['original_filename']}"
                )
            else:
                log.warning("Missing original_filename!")

    log.info("=" * 80)
    log.info("SUMMARY:")
    log.info(f"Total unique files: {len(by_filename)}")
    log.info(f"Documents with 'unknown' filename: {unknown_count}")
    log.info(f"Documents with 'tmp' filename: {temp_count}")

    if unknown_count > 0 or temp_count > 0:
        log.warning("Some documents have incorrect filenames!")
        log.warning("This means the metadata update didn't work properly.")
        log.warning("Recommended action: Clear database and re-index files.")
    else:
        log.info("All documents have proper filenames!")

    log.info("=" * 80)

    # Show detailed metadata for first 3 documents
    log.info("DETAILED METADATA (first 3 documents):")
    log.info("-" * 80)
    for i, doc in enumerate(all_docs[:3]):
        log.info(f"Document {i+1}:")
        log.info(f"ID: {doc.id}")
        log.info(f"Content preview: {doc.content[:100] if doc.content else ''}...")
        log.info("Metadata:")
        log.info(json.dumps(doc.meta, indent=2))

    return {
        "total_docs": len(all_docs),
        "unique_files": len(by_filename),
        "unknown_count": unknown_count,
        "temp_count": temp_count,
        "files": list(by_filename.keys()),
    }


if __name__ == "__main__":
    try:
        result = diagnose_documents()
        log.info("Returned data:", json.dumps(result, indent=2))
    except Exception as e:
        log.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
