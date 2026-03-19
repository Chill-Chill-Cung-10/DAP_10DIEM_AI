"""Vietnamese text normalisation and canonicalisation utilities."""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

_PREFIX_PATTERNS = tuple(re.compile(p) for p in (
    r"^cho\s+(toi|minh|em|anh|chi|ad|admin)\s+(hoi|biet)\s+",
    r"^xin\s+(hoi|cho\s+biet|tu\s+van)\s+",
    r"^toi\s+muon\s+(hoi|biet)\s+",
    r"^minh\s+muon\s+(hoi|biet)\s+",
    r"^giai\s+thich\s+cho\s+(toi|minh|em|anh|chi)\s+",
    r"^tu\s+van\s+giu?p\s+(toi|minh|em|anh|chi)\s+",
    r"^cho\s+(toi|minh|em|anh|chi)\s+hoi\s+ve\s+",
    r"^quy\s+dinh\s+(phap\s+luat\s+)?ve\s+",
    r"^theo\s+quy\s+dinh\s+(cua\s+)?phap\s+luat\s+(viet\s+nam\s+)?ve\s+",
    r"^theo\s+luat\s+(viet\s+nam\s+)?ve\s+",
    r"^thong\s+tin\s+ve\s+",
    r"^noi\s+ve\s+",
))

_SUFFIX_PATTERNS = tuple(re.compile(p) for p in (
    r"\s+la\s+gi\s*$",
    r"\s+la\s+sao\s*$",
    r"\s+nhu\s+the\s+nao\s*$",
    r"\s+ra\s+sao\s*$",
    r"\s+duoc\s+khong\s*$",
    r"\s+khong\s*$",
    r"\s+ntn\s*$",
    r"\s+nhe\s*$",
    r"\s+a\s*$",
    r"\s+ah\s*$",
))

_NOISE_PATTERNS = tuple(re.compile(p) for p in (
    r"\bphap\s+luat\b",
    r"\bquy\s+dinh\b",
    r"\btai\s+viet\s+nam\b",
    r"\bo\s+viet\s+nam\b",
    r"\bve\s+viec\b",
    r"\blien\s+quan\s+den\b",
    r"\bdoi\s+voi\b",
    r"\btrong\s+truong\s+hop\b",
    r"\bcho\s+biet\b",
    r"\bxin\s+hoi\b",
    r"\btu\s+van\b",
    r"\bgiai\s+thich\b",
))

_STOPWORDS = {
    "cho", "toi", "minh", "em", "anh", "chi", "biet", "hoi",
    "xin", "tu", "van", "giai", "thich", "giup", "dum", "nhe",
    "a", "ah", "la", "gi", "nhu", "the", "nao", "ra", "sao",
    "duoc", "khong", "ve", "tai", "o", "viet", "nam", "theo",
    "cua", "noi", "thong", "tin", "lien", "quan", "den", "doi",
    "voi", "trong", "truong", "hop", "viec",
}

_QUESTION_PREFIX_RE = re.compile(
    r"^\s*câu\s*hỏi\s*\d+\s*:\s*", flags=re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_question_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = _QUESTION_PREFIX_RE.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def canonicalize_question_text(text: str) -> str:
    text = normalize_question_text(text)
    text = strip_vietnamese_accents(text)
    text = text.replace("ntn", "nhu the nao")

    for pattern in _PREFIX_PATTERNS:
        text = pattern.sub("", text)
    for pattern in _SUFFIX_PATTERNS:
        text = pattern.sub("", text)
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)