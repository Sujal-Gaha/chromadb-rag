from haystack_integrations.document_stores.chroma import ChromaDocumentStore
import json


def diagnose_documents():
    print("=" * 80)
    print("CHROMADB DOCUMENT METADATA DIAGNOSTIC")
    print("=" * 80)

    document_store = ChromaDocumentStore(
        collection_name="document_collection",
        persist_path="./chroma_data",
    )

    # Get all documents
    all_docs = document_store.filter_documents()

    print(f"\nTotal documents in database: {len(all_docs)}\n")

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

    # Print summary
    print("FILENAME DISTRIBUTION:")
    print("-" * 80)
    for filename, docs in sorted(by_filename.items()):
        print(f"\n📄 {filename}")
        print(f"   Chunks: {len(docs)}")

        # Show first chunk's metadata
        if docs:
            first_doc = docs[0]
            print(f"   First chunk ID: {first_doc['doc_id']}")
            print(f"   Content: {first_doc['content_preview']}")
            print(f"   Metadata keys: {list(first_doc['meta'].keys())}")

            # Check for original_filename
            if "original_filename" in first_doc["meta"]:
                print(
                    f"   ✓ Has original_filename: {first_doc['meta']['original_filename']}"
                )
            else:
                print("   ✗ Missing original_filename!")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total unique files: {len(by_filename)}")
    print(f"Documents with 'unknown' filename: {unknown_count}")
    print(f"Documents with 'tmp' filename: {temp_count}")

    if unknown_count > 0 or temp_count > 0:
        print("\n⚠ WARNING: Some documents have incorrect filenames!")
        print("   This means the metadata update didn't work properly.")
        print("   Recommended action: Clear database and re-index files.")
    else:
        print("\n✓ All documents have proper filenames!")

    print("=" * 80)

    # Show detailed metadata for first 3 documents
    print("\nDETAILED METADATA (first 3 documents):")
    print("-" * 80)
    for i, doc in enumerate(all_docs[:3]):
        print(f"\nDocument {i+1}:")
        print(f"ID: {doc.id}")
        print(f"Content preview: {doc.content[:100] if doc.content else ''}...")
        print("Metadata:")
        print(json.dumps(doc.meta, indent=2))

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
        print("\n\nReturned data:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
