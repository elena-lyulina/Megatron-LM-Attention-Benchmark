import re
import unicodedata
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from langdetect import detect, LangDetectException


# ── 1. Subject filter ─────────────────────────────────────────────────────────
# subjects field from manu/project_gutenberg dataset is a list of strings
_EXCLUDED_SUBJECTS = {"poetry", "poems", "verse", "ballads", "sonnets"}

def has_literature_subject(subjects: list[str]) -> bool:
    return not any(
        any(excl in s.lower() for excl in _EXCLUDED_SUBJECTS)
        for s in subjects
    )


# ── 2. Duplicate filtering ────────────────────────────────────────────────────
# TODO: deduplicate by title/author similarity across editions


# ── 3. Gutenberg boilerplate stripping ───────────────────────────────────────
_START_RE = re.compile(r'\*\*\* ?START OF .+?\*\*\*', re.IGNORECASE)
_END_RE   = re.compile(r'\*\*\* ?END OF .+?\*\*\*',   re.IGNORECASE)

def strip_gutenberg_boilerplate(text: str) -> Optional[str]:
    start = _START_RE.search(text)
    end   = _END_RE.search(text)
    if start is None or end is None:
        return None  # malformed file, discard
    return text[start.end():end.start()].strip()


# ── 4. Language detection ─────────────────────────────────────────────────────
def is_english(text: str, sample_chars: int = 5_000) -> bool:
    try:
        return detect(text[:sample_chars]) == 'en'
    except LangDetectException:
        return False


# ── 5. NeMo-style normalization ───────────────────────────────────────────────
_URL_RE          = re.compile(r'https?://\S+|www\.\S+')
_HTML_RE         = re.compile(r'<[^>]+>')
_MULTI_BLANK_RE  = re.compile(r'\n{3,}')

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = _URL_RE.sub('', text)
    text = _HTML_RE.sub('', text)
    text = _MULTI_BLANK_RE.sub('\n\n', text)
    return text.strip()


# ── 6. Chunking ───────────────────────────────────────────────────────────────
# ~500 tokens ≈ 2000 chars (assuming ~4 chars/token)
def split_into_chunks(text: str, target_chars: int = 2_000) -> list[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        if current and current_len + len(para) > target_chars:
            chunks.append('\n\n'.join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


# ── 7. Heuristic metrics ──────────────────────────────────────────────────────

def _alphabetic_ratio(text: str) -> float:
    return sum(c.isalpha() for c in text) / len(text) if text else 0.0


def _numeric_density(text: str) -> float:
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


_SPECIAL_CHARS_RE = re.compile(r'[|+={}\[\]<>#@$%^~\\]')

def _special_char_density(text: str) -> float:
    return len(_SPECIAL_CHARS_RE.findall(text)) / len(text) if text else 0.0


def _short_line_ratio(text: str, min_len: int = 40) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    return sum(1 for l in lines if len(l.strip()) < min_len) / len(lines) if lines else 1.0


def _line_length_cv(text: str) -> float:
    # tables/lists → uniform line lengths → low CV; prose → high CV
    lengths = [len(l.strip()) for l in text.splitlines() if l.strip()]
    if len(lengths) < 2:
        return 0.0
    mean = statistics.mean(lengths)
    return statistics.stdev(lengths) / mean if mean > 0 else 0.0


def _type_to_token_ratio(text: str) -> float:
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0


def _repetition_score(text: str, n: int = 5) -> float:
    # fraction of n-grams that are duplicates
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams  = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts  = Counter(ngrams)
    duplicates = sum(c - 1 for c in counts.values())
    return duplicates / len(ngrams)


# ── 8. Heuristic filter ───────────────────────────────────────────────────────

@dataclass
class HeuristicFilter:
    alphabetic_ratio_min:    float = 0.70
    numeric_density_max:     float = 0.15
    special_char_density_max: float = 0.05
    short_line_ratio_max:    float = 0.50
    line_length_cv_min:      float = 0.20
    ttr_min:                 float = 0.30
    repetition_score_max:    float = 0.20

    def scores(self, text: str) -> dict[str, float]:
        return {
            'alphabetic_ratio':    _alphabetic_ratio(text),
            'numeric_density':     _numeric_density(text),
            'special_char_density': _special_char_density(text),
            'short_line_ratio':    _short_line_ratio(text),
            'line_length_cv':      _line_length_cv(text),
            'ttr':                 _type_to_token_ratio(text),
            'repetition_score':    _repetition_score(text),
        }

    def passes(self, text: str) -> bool:
        s = self.scores(text)
        return (
            s['alphabetic_ratio']     >= self.alphabetic_ratio_min
            and s['numeric_density']      <= self.numeric_density_max
            and s['special_char_density'] <= self.special_char_density_max
            and s['short_line_ratio']     <= self.short_line_ratio_max
            and s['line_length_cv']       >= self.line_length_cv_min
            and s['ttr']                  >= self.ttr_min
            and s['repetition_score']     <= self.repetition_score_max
        )


# ── 9. Filter chunks and merge consecutive clean ones ─────────────────────────

def filter_and_merge_chunks(chunks: list[str], heuristic_filter: HeuristicFilter) -> list[str]:
    # returns each contiguous run of passing chunks merged into one string
    runs, current_run = [], []
    for chunk in chunks:
        if heuristic_filter.passes(chunk):
            current_run.append(chunk)
        else:
            if current_run:
                runs.append('\n\n'.join(current_run))
                current_run = []
    if current_run:
        runs.append('\n\n'.join(current_run))
    return runs


# ── 10. 8k token sequence selection ──────────────────────────────────────────
# TODO: select random 8190-token sequence from runs that are long enough