"""Corpus ingestion and chunking.

Documents live in data/corpus/ as plain text or markdown, one file per source, with a
front-matter line `doc_id: <id>`. Section-aware chunking relies on regulation-style headings
such as "§ 1005.11 Procedures for resolving errors." and paragraph markers "(a)", "(b)(1)".

Heading styles recognized:
  * CFR sections        "§ 1005.11 ..."  (Reg E) and "§ 1026.13 ..." (Reg Z)
  * Interpretations     "Section 1005.11—Procedures for Resolving Errors" (Supplement I)
  * Network guides      "Condition 13.6 ..." / "Reason Code 10.4 ..." (Visa dispute guidelines)

How references are attached (v6). Every line of a document is annotated with the reference in
force at that point: the current section, its lettered paragraph "(c)" and, beneath that, a
numbered sub-paragraph "(c)(1)". Both chunking strategies then label each chunk with every
reference whose text it contains (`section_refs`) and with the first of them (`section_ref`).
Earlier versions located fixed-size chunks by searching for their text in the original document;
because chunk text is whitespace-normalized that search usually failed, so most fixed chunks
(Configs A and B) carried no reference at all and could never score a retrieval hit.

Documents without recognizable headings fall back to fixed-size chunks with no reference, which
lowers retrieval-recall scoring for that document — so keep headings intact when building the
corpus.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import ChunkingCfg
from .refs import normalize_ref
from .schemas import Chunk

SECTION_RE = re.compile(
    r"^\s*(?:§\s*(?P<cfr>\d{4}\.\d+)|Section\s+(?P<interp>\d{4}\.\d+)"
    r"|(?:Condition|Reason\s+Code)\s+(?P<net>\d+\.\d+))(?![\d(])\s*[—–:.-]?\s*(?P<title>.*)$",
    re.M,
)
# A paragraph marker at the start of a line, possibly followed by a nested marker on the same line,
# e.g. "(b) Limitations ..." or "(b)(1) Timely notice ...".
PARA_LINE_RE = re.compile(r"^\s*\((?P<p1>[a-z0-9]{1,4})\)(?:\((?P<p2>[a-z0-9]{1,4})\))?\s")
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}

WORDS_PER_TOKEN = 1 / 1.3  # chunk sizes are configured in approximate tokens


def _ref_of(match: re.Match) -> str:
    return match.group("cfr") or match.group("interp") or match.group("net")


def load_documents(corpus_dir: Path) -> dict[str, str]:
    docs = {}
    for p in sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt")):
        if p.name.lower() == "readme.md":
            continue
        text = p.read_text(encoding="utf-8")
        m = re.match(r"doc_id:\s*(\S+)\s*\n", text)
        doc_id = m.group(1) if m else p.stem
        docs[doc_id] = text[m.end():] if m else text
    return docs


def _next_letter(cur: str | None, cand: str) -> bool:
    """'(i)' straight after paragraph '(h)' is the next lettered paragraph, not a roman numeral."""
    return bool(cur) and len(cur) == 1 and len(cand) == 1 and ord(cand) == ord(cur) + 1


def annotate_lines(text: str) -> list[tuple[str, str | None, str | None]]:
    """Return (line, paragraph_ref, finest_ref) for every non-empty line.

    paragraph_ref is section + lettered paragraph, e.g. "1005.11(c)" — the unit section-aware
    chunks are cut on. finest_ref adds one more level, e.g. "1005.11(c)(1)".
    """
    out = []
    sec = letter = num = None
    for line in text.splitlines():
        if not line.strip():
            continue
        h = SECTION_RE.match(line)
        if h:
            sec, letter, num = _ref_of(h), None, None
        elif sec and (pm := PARA_LINE_RE.match(line)):
            p1, p2 = pm.group("p1").lower(), pm.group("p2")
            if p1.isdigit():                      # "(1)" — sub-paragraph of the current letter
                num = p1 if letter else None
            elif p1 in ROMAN and num and not _next_letter(letter, p1):
                pass                              # "(i)" under "(b)(1)" — third level, not indexed
            elif p1.isalpha():                    # "(c)" — new lettered paragraph
                letter, num = p1, (p2.lower() if p2 else None)
        para = f"{sec}({letter})" if sec and letter else sec
        finest = f"{para}({num})" if para and letter and num else para
        out.append((line, para, finest))
    return out


def corpus_section_index(docs: dict[str, str]) -> set[str]:
    """Every normalized reference that exists in the corpus, at section, paragraph and
    sub-paragraph level. A citation not in this set is scored as fabricated (RQ2)."""
    index: set[str] = set()
    for text in docs.values():
        for _, para, finest in annotate_lines(text):
            for r in (para, finest):
                if r and (n := normalize_ref(r)):
                    index.add(n)
                    index.add(n.split("(")[0])
    return index


def _unique(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _windows(n_words: int, cfg: ChunkingCfg):
    size = max(1, int(cfg.chunk_size * WORDS_PER_TOKEN))
    step = max(1, size - int(cfg.overlap * WORDS_PER_TOKEN))
    for start in range(0, n_words, step):
        yield start, min(start + size, n_words)
        if start + size >= n_words:
            break


def _fixed_from_annotated(doc_id: str, annotated, cfg: ChunkingCfg, prefix: str) -> list[Chunk]:
    words, refs = [], []
    for line, _, finest in annotated:
        for w in line.split():
            words.append(w)
            refs.append(finest)
    chunks = []
    for i, (s, e) in enumerate(_windows(len(words), cfg)):
        chunk_refs = _unique(refs[s:e])
        chunks.append(Chunk(chunk_id=f"{prefix}__{i}", doc_id=doc_id,
                            section_ref=chunk_refs[0] if chunk_refs else None,
                            section_refs=chunk_refs, text=" ".join(words[s:e])))
    return chunks


def chunk_fixed(doc_id: str, text: str, cfg: ChunkingCfg) -> list[Chunk]:
    """Fixed windows of ~chunk_size tokens with overlap, labelled with every reference inside."""
    return _fixed_from_annotated(doc_id, annotate_lines(text), cfg, f"{doc_id}__fixed")


def chunk_section(doc_id: str, text: str, cfg: ChunkingCfg) -> list[Chunk]:
    """One chunk per lettered paragraph (e.g. 1005.11(c)); paragraphs longer than 1.5 x
    chunk_size are split into fixed windows that keep the paragraph reference."""
    annotated = annotate_lines(text)
    groups: list[tuple[str | None, list]] = []
    for item in annotated:
        para = item[1]
        if groups and groups[-1][0] == para:
            groups[-1][1].append(item)
        else:
            groups.append((para, [item]))
    if not any(p for p, _ in groups):
        return chunk_fixed(doc_id, text, cfg)

    chunks = []
    for j, (para, items) in enumerate(groups):
        body = "\n".join(line for line, _, _ in items).strip()
        if not body:
            continue
        n_words = len(body.split())
        if n_words > cfg.chunk_size * WORDS_PER_TOKEN * 1.5:
            for c in _fixed_from_annotated(doc_id, items, cfg, f"{doc_id}__{para}__{j}"):
                c.section_ref = para
                c.section_refs = _unique([para] + c.section_refs)
                chunks.append(c)
        else:
            refs = _unique([para] + [f for _, _, f in items])
            chunks.append(Chunk(chunk_id=f"{doc_id}__{para}__{j}", doc_id=doc_id,
                                section_ref=para, section_refs=refs, text=body))
    return chunks


def build_chunks(corpus_dir: Path, cfg: ChunkingCfg) -> list[Chunk]:
    docs = load_documents(corpus_dir)
    fn = chunk_section if cfg.strategy == "section" else chunk_fixed
    out = []
    for doc_id, text in docs.items():
        out.extend(fn(doc_id, text, cfg))
    return out
