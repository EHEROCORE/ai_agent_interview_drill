"""Robust text decoding + mojibake detection for ingestion source quality.

A RAG system is only as good as the text it indexes. Chinese corpora frequently arrive
as UTF-8/UTF-8-BOM/GBK or already-corrupted (mojibake). This module decodes bytes
robustly and flags suspicious text so bad sources can be caught before they pollute the
index — a frequently-overlooked but high-leverage engineering point.
"""

from __future__ import annotations

import re

_REPLACEMENT = "�"  # U+FFFD REPLACEMENT CHARACTER
_CJK_RE = re.compile(r"[一-鿿]")
# CP1252 "smart punctuation" codepoints that UTF-8 continuation bytes turn into when
# mis-decoded as CP1252 (en/em dash, curly quotes, dagger, bullet, ellipsis, ...).
_CP1252_ARTIFACTS = {
    0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
    0x2020, 0x2021, 0x2022, 0x2026, 0x2030, 0x2039, 0x203A,
}


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode bytes to text, returning ``(text, detected_encoding)``.

    Tries UTF-8 BOM, UTF-8, GB18030, then Latin-1 as a last resort. The detected
    encoding is returned so callers can record source provenance.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def is_mojibake(text: str) -> bool:
    """Heuristically detect corrupted/garbled text.

    Strong signals: the U+FFFD replacement character, or a dense run of Latin-1
    supplement / CP1252 punctuation artifacts with no cleanly-decoded CJK present.
    """
    if _REPLACEMENT in text:
        return True
    if _CJK_RE.search(text):
        # Cleanly decoded CJK is present -> the text is very likely fine.
        return False
    artifacts = sum(1 for ch in text if 0x80 <= ord(ch) <= 0xFF or ord(ch) in _CP1252_ARTIFACTS)
    return artifacts >= 3


def clean_text(text: str) -> str:
    """Normalize newlines/whitespace and strip stray replacement characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(_REPLACEMENT, "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()
