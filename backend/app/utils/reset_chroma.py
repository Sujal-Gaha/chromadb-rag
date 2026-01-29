import shutil
import os
import sys


def reset_chroma():
    chroma_paths = ["./chroma_data", "/data/chroma_db"]

    for path in chroma_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"✓ Deleted: {path}")
        else:
            print(f"✓ No existing data at: {path}")

    print("\nChromaDB reset complete. Restart your application.")


if __name__ == "__main__":
    confirm = input("This will delete all indexed documents. Continue? (y/n): ")
    if confirm.lower() == "y":
        reset_chroma()
    else:
        print("Cancelled.")
