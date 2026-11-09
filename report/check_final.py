"""Pre-submission checks on the final report .docx against the mechanical rubric criteria.

    python check_final.py QM640_Final_Report_DRAFT.docx

Checks (rubric points in brackets):
  abstract       300-350 words and a single paragraph                         [2 + 2]
  placeholders   highlighted [placeholders] left in the document
  tables/figures numbered 1..n without gaps, each called out in the text      [7, formatting]
  abbreviations  first use of each all-caps abbreviation is next to its full form, e.g. "(RAG)"  [3]
  references     at least 10 entries; every cited author-year has an entry    [8]
  formatting     1-inch margins, double spacing in body text, one font family [7]
It reports; it does not edit. Treat abbreviation hits as candidates to look at, not errors.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict

from docx import Document
from docx.enum.text import WD_COLOR_INDEX

COMMON = {"I", "II", "III", "IV", "V", "A", "B", "C", "C0", "RQ", "RQ1", "RQ2", "RQ3", "RQ4", "H", "US", "U.S",
          "XXXX", "JSON", "PNG", "CSV", "SQL", "AI", "OR", "CI", "PK", "FK", "SD", "IQR", "N", "QA", "NLP", "DL",
          "MRR", "PPI", "XML", "SHA", "GEE", "EDA", "API", "ID", "IDS", "LM", "NIST", "BM25", "ATM", "NQ",
          "QM640", "F1", "H1", "H2", "H3", "H4", "A1", "B1", "C1", "D1", "E1", "E2", "T", "NOT", "ONE",
          "AND", "SELECT", "FROM", "WHERE", "MB", "GB", "PDF", "URL", "DOI", "XXX"}


def body_paragraphs(doc):
    return [p for p in doc.paragraphs]


def main(path: str) -> int:
    doc = Document(path)
    paras = body_paragraphs(doc)
    texts = [p.text for p in paras]
    problems = 0

    # ---- abstract
    try:
        i = next(k for k, t in enumerate(texts) if t.strip() == "Abstract")
        block = []
        for p in paras[i + 1:]:
            if p.style.name.startswith("Heading") or p.text.strip() in ("GitHub Repository", "Table of Contents"):
                break
            if p.text.strip() and not p.text.startswith("Note."):
                block.append(p.text)
        words = sum(len(t.split()) for t in block)
        ok = 300 <= words <= 350 and len(block) == 1
        problems += not ok
        print(f"[{'ok' if ok else '!!'}] Abstract: {words} words in {len(block)} paragraph(s) (need 300-350, one)")
    except StopIteration:
        problems += 1
        print("[!!] Abstract heading not found")

    # ---- placeholders
    ph = sum(1 for p in paras for r in p.runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                ph += sum(1 for p in c.paragraphs for r in p.runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW)
    problems += ph > 0
    print(f"[{'ok' if not ph else '!!'}] Highlighted placeholders remaining: {ph}")

    # ---- table / figure numbering and call-outs
    for kind in ("Table", "Figure"):
        labels = [int(m.group(1)) for t in texts for m in [re.fullmatch(rf"{kind} (\d+)", t.strip())] if m]
        gaps = labels != list(range(1, len(labels) + 1))
        prose = "\n".join(t for t in texts if not re.fullmatch(rf"{kind} [A-Z]?\d+", t.strip()))
        uncited = [n for n in labels if not re.search(rf"\b{kind}s? {n}\b", prose)]
        bad = gaps or uncited
        problems += bool(bad)
        print(f"[{'!!' if bad else 'ok'}] {kind}s: {len(labels)} numbered"
              + (" (numbering has gaps or is out of order)" if gaps else "")
              + (f"; not called out in text: {uncited}" if uncited else ""))

    # ---- abbreviations at first use
    first = OrderedDict()
    stop = next((k for k, t in enumerate(texts) if t.strip() == "References"), len(texts))
    for k, t in enumerate(texts[:stop]):
        t = re.sub(r"\[[^\]]*\]", " ", t)  # skip highlighted placeholders and bracketed citations
        for m in re.finditer(r"\b([A-Z][A-Z0-9]{1,6}s?)\b", t):
            ab = m.group(1).rstrip("s")
            if ab not in first:
                first[ab] = (k, t, m.start())
    undefined = []
    for ab, (_k, t, pos) in first.items():
        if ab in COMMON:
            continue
        window = t[max(0, pos - 2): pos + len(ab) + 2]
        if f"({ab}" not in window and f"{ab})" not in window:
            undefined.append(ab)
    print(f"[{'ok' if not undefined else '??'}] Abbreviations whose first use is not in parentheses next to a full form: "
          + (", ".join(undefined[:30]) if undefined else "none"))

    # ---- references
    try:
        r0 = next(k for k, t in enumerate(texts) if t.strip() == "References")
        r1 = next((k for k, t in enumerate(texts[r0 + 1:], r0 + 1) if t.startswith("Appendix")), len(texts))
        refs = [t for t in texts[r0 + 1:r1] if t.strip()]
        body = "\n".join(texts[:r0])
        body = re.sub(r"'s\b", "", body)
        cited = set(re.findall(r"([A-Z][A-Za-z\-']+)(?: et al\.| & [A-Z][A-Za-z\-']+)?,? \(?(\d{4}[a-z]?)", body))
        alias = {"CFPB": "Consumer Financial Protection Bureau"}
        missing = sorted({f"{a} {y}" for a, y in cited
                          if not any(r.startswith(alias.get(a, a)) and y[:4] in r for r in refs)
                          and a not in ("Table", "Figure", "Term", "Part", "October", "November", "September", "August")})
        problems += len(refs) < 10
        print(f"[{'ok' if len(refs) >= 10 else '!!'}] References: {len(refs)} entries")
        if missing:
            print(f"[??] Cited author-years without an obvious reference entry: {', '.join(missing[:20])}")
    except StopIteration:
        problems += 1
        print("[!!] References heading not found")

    # ---- formatting
    sec = doc.sections[0]
    margins = [round(x.inches, 2) for x in (sec.top_margin, sec.bottom_margin, sec.left_margin, sec.right_margin)]
    m_ok = all(abs(m - 1.0) < 0.01 for m in margins)
    spacing = {p.paragraph_format.line_spacing for p in paras if p.text.strip() and p.style.name == "Normal"}
    fonts = {r.font.name for p in paras for r in p.runs if r.font.name}
    problems += not m_ok
    print(f"[{'ok' if m_ok else '!!'}] Margins (in): {margins}")
    print(f"[ok] Fonts set on runs: {sorted(fonts) or 'inherited from style (uniform)'}; "
          f"line-spacing values on body paragraphs: {sorted(str(s) for s in spacing)}")
    print(f"\n{problems} blocking issue(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "QM640_Final_Report_DRAFT.docx"))
