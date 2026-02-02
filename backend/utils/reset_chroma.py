import shutil
import os

from .logger import get_logger

log = get_logger(__name__)


def reset_chroma():
    chroma_paths = ["./chroma_data", "/data/chroma_db"]

    for path in chroma_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            log.info(f"Deleted: {path}")
        else:
            log.info(f"No existing data at: {path}")

    log.info("ChromaDB reset complete. Restart your application.")


if __name__ == "__main__":
    confirm = input("This will delete all indexed documents. Continue? (y/n): ")
    if confirm.lower() == "y":
        reset_chroma()
    else:
        log.info("Cancelled.")
