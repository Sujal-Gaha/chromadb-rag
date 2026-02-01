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
    @staticmethod
    def extract_title_from_content(content: str) -> Optional[str]:
        title_match = re.search(r"\*\*Title: (.+?)\*\*", content[:500])
        if title_match:
            return title_match.group(1).strip()

        lines = content.split("\n")
        for line in lines[:10]:
            if "Title:" in line:
                return line.replace("Title:", "").strip()

        return None

    @staticmethod
    def convert_title_to_filename(title: str) -> str:
        filename = title.lower()

        filename = filename.replace(" ", "-")

        filename = re.sub(r"[^\w\-\.]", "", filename)

        if not filename.endswith(".txt"):
            filename += ".txt"

        return filename

    @staticmethod
    def map_temp_to_original(temp_filename: str, content: str) -> str:
        title = FilenameMapper.extract_title_from_content(content)
        if title:
            original_filename = FilenameMapper.convert_title_to_filename(title)

            for known_filename in ORIGINAL_FILENAMES:
                if (
                    known_filename in original_filename
                    or original_filename in known_filename
                ):
                    return known_filename

            return original_filename

        if temp_filename.startswith("tmp"):
            temp_id = temp_filename.replace("tmp", "").replace(".txt", "")
            return f"document-{temp_id}.txt"

        return temp_filename

    @staticmethod
    def is_temp_filename(filename: str) -> bool:
        return filename.startswith("tmp") and filename.endswith(".txt")

    @staticmethod
    def normalize_filename(filename: str) -> str:
        if "/" in filename:
            filename = filename.split("/")[-1]

        if not filename.endswith(".txt"):
            filename += ".txt"

        return filename.lower()
