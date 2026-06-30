"""Semantic-aware document chunking for medical device knowledge bases.

Uses domain-specific separators (device codes, section headers, bullet points)
instead of naive character splitting. This preserves semantic coherence of
technical documentation like fault trees and maintenance procedures.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.shared.config import settings


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Create a chunker tuned for medical device documentation.

    Domain-aware separators preserve structure:
      1. Device code boundaries (e.g. "MED-VENT-X200")
      2. Section headers (e.g. "### 故障诊断")
      3. Bullet/item markers (e.g. "①", "•", "-")
      4. Paragraph / sentence boundaries
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n## ",      # H2 section header
            "\n# ",       # H1 section header
            "\n\n",       # Paragraph break
            "\n",         # Line break
            "；",         # Chinese semicolon
            "；",         # Full-width semicolon
            " ",          # Space
            "",           # Character-level split as last resort
        ],
        length_function=len,
    )


def chunk_by_device(text: str) -> list[dict]:
    """Split text into device-specific sections.

    Useful when a single document contains multiple device manuals.
    Returns a list of dicts with 'device_code' and 'content' keys.
    """
    # Pattern: detect device codes like "MED-VENT-X200"
    import re
    device_pattern = re.compile(r"(MED-[A-Z]+-\S+)")
    parts = device_pattern.split(text)

    result = []
    current_device = None
    current_content = []

    for part in parts:
        match = device_pattern.match(part.strip())
        if match:
            if current_device and current_content:
                result.append({
                    "device_code": current_device,
                    "content": "\n".join(current_content),
                })
            current_device = match.group(1)
            current_content = []
        else:
            current_content.append(part)

    if current_device and current_content:
        result.append({
            "device_code": current_device,
            "content": "\n".join(current_content),
        })

    return result
