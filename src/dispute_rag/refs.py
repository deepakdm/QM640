"""Section-reference normalization and matching.

Every groundedness and driver measure in the study (fabricated citation, correct clause cited,
clause in context, retrieval recall) is a comparison between section references written in
different ways: the model may write "§ 1005.11(c)", "12 CFR 1005.11(c)" or "1005.11 (c)", the gold
label says "1005.11(c)(1)", and a chunk carries "1005.11(c)". The rules below make those
comparisons deterministic and identical across conditions, so they belong in the Methods section:

  * normalize:   strip "§", "12 CFR", "C.F.R.", "section", "sec.", spaces, brackets and trailing
                 punctuation; lower-case.
  * cites gold:  a citation matches a gold clause if it IS the gold clause or a MORE SPECIFIC
                 sub-paragraph of it (cited 1005.6(b)(1) matches gold 1005.6(b)). Citing only the
                 parent section (1005.6) does not match a sub-paragraph gold clause.
  * in context:  a gold clause is in context if a retrieved passage carries that clause, one of its
                 sub-paragraphs, or a parent of it (a passage labelled 1005.6(b) contains (b)(1)).
  * fabricated:  a recognizable reference that does not exist in the corpus section index. The
                 index records sections and two levels of sub-paragraph, so deeper citations are
                 checked at that depth. Citations with no recognizable reference ("Regulation E")
                 are counted separately as unverifiable, not as fabricated.
"""
from __future__ import annotations

import re
from typing import Iterable

_PREFIX_RE = re.compile(
    r"(§+|12\s*c\.?\s*f\.?\s*r\.?\s*(part|pt\.?)?|c\.?\s*f\.?\s*r\.?|section|sec\.|reg(ulation)?\s*[ez]|"
    r"reason\s*code|condition)",
    re.I,
)
# A CFR reference (1005.11, 1005.11(c)(1)) or a network reference (13.1, 10.4).
_REF_RE = re.compile(r"(?P<base>\d{4}\.\d+[a-z]?|\d{1,2}\.\d{1,2})(?P<paras>(\([a-z0-9]+\))*)", re.I)
_PARA_RE = re.compile(r"\(([a-z0-9]+)\)", re.I)

INDEX_DEPTH = 2  # sub-paragraph levels recorded in the corpus index, e.g. 1005.6(b)(1)


def normalize_ref(ref: str | None) -> str | None:
    """Canonical form of a section reference, or None if none can be recognized."""
    if not ref:
        return None
    s = _PREFIX_RE.sub(" ", str(ref))
    s = re.sub(r"[\[\]\s]", "", s).strip(".,;:").lower()
    m = _REF_RE.search(s)
    if not m:
        return None
    return m.group("base") + (m.group("paras") or "")


def split_ref(ref: str) -> tuple[str, list[str]]:
    """'1005.6(b)(1)' -> ('1005.6', ['b', '1'])."""
    m = _REF_RE.match(ref)
    if not m:
        return ref, []
    return m.group("base"), _PARA_RE.findall(m.group("paras") or "")


def truncate(ref: str, depth: int = INDEX_DEPTH) -> str:
    base, paras = split_ref(ref)
    return base + "".join(f"({p})" for p in paras[:depth])


def is_descendant_or_equal(child: str, parent: str) -> bool:
    """True if `child` is `parent` or a more specific sub-paragraph of it."""
    return child == parent or child.startswith(parent + "(")


def cites_gold(cited: Iterable[str], gold: Iterable[str]) -> bool:
    c = [n for n in (normalize_ref(x) for x in cited) if n]
    g = [n for n in (normalize_ref(x) for x in gold) if n]
    return any(is_descendant_or_equal(ci, gi) for ci in c for gi in g)


def related(a: str, b: str) -> bool:
    """Either reference contains the other."""
    return is_descendant_or_equal(a, b) or is_descendant_or_equal(b, a)


def clause_in_context(retrieved: Iterable[str], gold: Iterable[str]) -> bool:
    r = [n for n in (normalize_ref(x) for x in retrieved) if n]
    g = [n for n in (normalize_ref(x) for x in gold) if n]
    return any(related(ri, gi) for ri in r for gi in g)


def classify_citations(cited: Iterable[str], corpus_index: set[str]) -> dict:
    """Split citations into valid, fabricated (recognizable but absent) and unverifiable."""
    valid, fabricated, unverifiable = [], [], []
    for raw in cited:
        n = normalize_ref(raw)
        if n is None:
            unverifiable.append(str(raw))
        elif truncate(n) in corpus_index:
            valid.append(n)
        else:
            fabricated.append(n)
    return {"valid": valid, "fabricated": fabricated, "unverifiable": unverifiable}
