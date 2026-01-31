import re
from typing import Optional

ORIGINAL_FILENAMES = [
    "the-absurd-adolescence-of-john-doe.txt",
    "the-calm-storm-of-john-doe.txt",
    "the-library-of-small-disappointments.txt",
    "the-last-book.txt",
    "the-ordinary-tuesday.txt",
    "the-quiet-clockwork-of-john-doe.txt",
    "the-uninvited-guest.txt",
    "the-wedding-that-wasnt.txt",
]


class FilenameMapper:
    """Maps temporary filenames to original filenames."""

    @staticmethod
    def extract_title_from_content(content: str) -> Optional[str]:
        """Extract title from document content."""
        # Pattern 1: **Title: Something**
        title_match = re.search(r"\*\*Title: (.+?)\*\*", content[:500])
        if title_match:
            return title_match.group(1).strip()

        # Pattern 2: Title: Something (without asterisks)
        lines = content.split("\n")
        for line in lines[:10]:
            if "Title:" in line:
                return line.replace("Title:", "").strip()

        return None

    @staticmethod
    def convert_title_to_filename(title: str) -> str:
        """Convert a title to standardized filename format."""
        # Convert to lowercase
        filename = title.lower()

        # Replace spaces with hyphens
        filename = filename.replace(" ", "-")

        # Remove special characters
        filename = re.sub(r"[^\w\-\.]", "", filename)

        # Ensure .txt extension
        if not filename.endswith(".txt"):
            filename += ".txt"

        return filename

    @staticmethod
    def map_temp_to_original(temp_filename: str, content: str) -> str:
        """
        Map a temporary filename to its original filename.

        Args:
            temp_filename: Temporary filename (e.g., tmpnmgwi1e0.txt)
            content: Document content to extract title from

        Returns:
            Original filename if found, otherwise returns the temp filename
        """
        # If it's already an original filename, return it
        if temp_filename in ORIGINAL_FILENAMES:
            return temp_filename

        # Try to extract title from content
        title = FilenameMapper.extract_title_from_content(content)
        if title:
            original_filename = FilenameMapper.convert_title_to_filename(title)

            # Check if this matches one of our known filenames
            for known_filename in ORIGINAL_FILENAMES:
                if (
                    known_filename in original_filename
                    or original_filename in known_filename
                ):
                    return known_filename

            return original_filename

        # Fallback: try to guess from temp filename pattern
        if temp_filename.startswith("tmp"):
            # Extract number or pattern and map to known files
            # This is a simple mapping - you might need to adjust based on your data
            temp_id = temp_filename.replace("tmp", "").replace(".txt", "")
            # You could maintain a mapping of temp IDs to original filenames
            # For now, return a placeholder
            return f"document-{temp_id}.txt"

        return temp_filename

    @staticmethod
    def is_temp_filename(filename: str) -> bool:
        """Check if a filename is temporary."""
        return filename.startswith("tmp") and filename.endswith(".txt")

    @staticmethod
    def normalize_filename(filename: str) -> str:
        """Normalize any filename to a standard format."""
        # Remove path if present
        if "/" in filename:
            filename = filename.split("/")[-1]

        # Ensure .txt extension
        if not filename.endswith(".txt"):
            filename += ".txt"

        return filename.lower()
