"""Step 0b: build the regulatory corpus in data/corpus/ (run once, on your own machine).

Regulation E and its official interpretations (Supplement I) are both published in the eCFR XML for
12 CFR Part 1005, so one download gives two corpus files:

    data/corpus/reg_e.md          § 1005.1 ... § 1005.36   (doc_id: reg_e)
    data/corpus/reg_e_interp.md   Supplement I to Part 1005 (doc_id: reg_e_interp)

Optional extras, from files you save yourself:
    --faq-html  saved copy of the CFPB "Electronic Fund Transfers FAQs" page  -> cfpb_eft_faq.md
    --visa-pdf  Visa "Dispute Management Guidelines for Visa Merchants" PDF   -> network_rules.md

Usage:
    python scripts/00b_build_corpus.py
    python scripts/00b_build_corpus.py --faq-html eft_faqs.html --visa-pdf visa_guide.pdf
    python scripts/00b_build_corpus.py --xml part1005.xml      # if you downloaded the XML by hand

Sources (record the date printed by this script in the report):
    eCFR API:  https://www.ecfr.gov/developers/documentation/api/v1
    Part 1005: https://www.ecfr.gov/current/title-12/chapter-X/part-1005
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ECFR = "https://www.ecfr.gov/api/versioner/v1"
OUT = Path("data/corpus")

p = argparse.ArgumentParser()
p.add_argument("--xml", help="local copy of the Part 1005 XML (skips the download)")
p.add_argument("--date", help="eCFR issue date YYYY-MM-DD (default: latest available)")
p.add_argument("--faq-html")
p.add_argument("--visa-pdf")
args = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def get(url: str) -> bytes:
    """Fetch a URL. The eCFR server answers 406 when the Accept header does not suit the endpoint,
    so try a few common values before giving up."""
    last = None
    for accept in ("*/*", "application/xml", "application/json", None):
        headers = {"User-Agent": "Mozilla/5.0 (QM640 capstone corpus builder)"}
        if accept:
            headers["Accept"] = accept
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (406, 415):
                break
    raise last


def text_of(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


# An eCFR paragraph often starts a sub-paragraph straight after its italic heading, e.g.
# "(a) Definition of error. (1) Except as provided ...". Split those so each marker starts a line,
# which is what corpus.py uses to attach references such as 1005.11(a)(1).
INLINE_SUB = re.compile(r"^(\([a-z]{1,2}\)\s+[^()]{1,160}?[.\u2014\u2013:-])\s*(\((?:\d{1,2}|[ivx]{1,4})\)\s)")


def split_inline(line: str) -> list[str]:
    m = INLINE_SUB.match(line)
    if not m:
        return [line]
    return [m.group(1), line[m.end(1):].strip()]


def section_lines(div) -> list[str]:
    lines = []
    for el in div.iter():
        tag = el.tag.upper()
        if tag == "HEAD":
            h = text_of(el)
            if re.match(r"^§\s*\d{4}\.\d+", h):
                lines.append("")
                lines.append(h)
        elif tag in ("P", "FP"):
            t = text_of(el)
            if t:
                lines.extend(split_inline(t))
    return lines


def appendix_lines(div) -> list[str]:
    lines = []
    for el in div.iter():
        tag = el.tag.upper()
        if tag in ("HEAD", "HD1", "HD2", "HD3"):
            h = text_of(el)
            # "Section 1005.11—Procedures for Resolving Errors" is recognized by corpus.py
            lines.extend(["", h])
        elif tag in ("P", "FP"):
            t = text_of(el)
            if t:
                lines.append(t)
    return lines


# ------------------------------------------------------------------ Regulation E + Supplement I
if args.xml:
    raw = Path(args.xml).read_bytes()
    issue = args.date or "local file"
else:
    from datetime import date as _date, timedelta as _td

    candidates = [args.date] if args.date else []
    if not candidates:
        try:
            t12 = next(t for t in json.loads(get(f"{ECFR}/titles.json"))["titles"] if t["number"] == 12)
            for k in ("latest_issue_date", "up_to_date_as_of", "latest_amended_on"):
                if t12.get(k) and t12[k] not in candidates:
                    candidates.append(t12[k])
        except Exception as e:  # titles.json unreachable: fall back to recent dates
            print(f"  could not read titles.json ({e}); trying recent dates")
        base = _date.fromisoformat(candidates[0]) if candidates else _date.today()
        candidates += [(base - _td(days=d)).isoformat() for d in (1, 2, 3, 7, 14, 30, 60)]
    raw, issue, tried = None, None, []
    for d in candidates:
        for url in (f"{ECFR}/full/{d}/title-12.xml?part=1005",
                    f"{ECFR}/full/{d}/title-12.xml?chapter=X&part=1005"):
            print(f"Trying {url}")
            try:
                raw, issue = get(url), d
                break
            except urllib.error.HTTPError as e:
                tried.append(f"{e.code} {url}")
                print(f"  -> HTTP {e.code}")
            except Exception as e:
                tried.append(f"{type(e).__name__} {url}")
                print(f"  -> {type(e).__name__}: {e}")
        if raw:
            break
    if not raw:
        raise SystemExit("\nCould not download Part 1005 from the eCFR API. Attempts:\n  " + "\n  ".join(tried)
                         + "\n\nUse the manual route: save the XML of 12 CFR Part 1005 from your browser as part1005.xml "
                           "in this folder, then run:\n  python scripts/00b_build_corpus.py --xml part1005.xml")
    print(f"Downloaded 12 CFR Part 1005 as of {issue}")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/raw/ecfr_part1005.xml").write_bytes(raw)

head = raw.lstrip(b"\xef\xbb\xbf \r\n\t")[:300]
if not head.startswith(b"<") or head.lower().startswith((b"<!doctype html", b"<html")) or not (b"<?XML" in head.upper() or b"<DIV" in head.upper() or b"<ECFR" in head.upper()) \
        or b'TYPE="SECTION"' not in raw.upper().replace(b"'", b'"').replace(b'TYPE = "', b'TYPE="'):
    shown = head.decode("utf-8", "replace").replace("\n", " ")
    raise SystemExit(
        "\nThis file is not the eCFR XML. It starts with:\n  " + shown[:250] + "\n\n"
        "Open the URL below in your browser; it must show XML text that begins with <?xml ... and contains DIV elements.\n"
        "Save that response as a file ending in .xml (in Chrome: right-click > Save as, or Cmd+S and choose 'Page Source'/'Webpage, HTML Only').\n"
        f"  {ECFR}/full/<date>/title-12.xml?part=1005\n"
        "If the browser shows an error or a JSON message instead, tell me exactly what it says.")
root = ET.fromstring(raw)
sections = [d for d in root.iter() if d.get("TYPE", "").upper() == "SECTION"]
appendices = [d for d in root.iter() if d.get("TYPE", "").upper() == "APPENDIX"]

reg_lines = []
for s in sections:
    reg_lines += section_lines(s)
supp = [a for a in appendices if "supplement i" in (a.get("N", "") + " " + text_of(a.find("HEAD") if a.find("HEAD") is not None else a)[:60]).lower()]
interp_lines = []
for a in supp:
    interp_lines += appendix_lines(a)

header = f"Source: 12 C.F.R. Part 1005 (eCFR, as of {issue})\n"
(OUT / "reg_e.md").write_text("doc_id: reg_e\n" + header + "\n".join(reg_lines).strip() + "\n", encoding="utf-8")
if interp_lines:
    (OUT / "reg_e_interp.md").write_text("doc_id: reg_e_interp\n" + header.replace("Part 1005", "Part 1005, Supplement I")
                                         + "\n".join(interp_lines).strip() + "\n", encoding="utf-8")

# ------------------------------------------------------------------ optional extras
if args.faq_html:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(Path(args.faq_html).read_text(encoding="utf-8", errors="ignore"), "html.parser")
    main = soup.find("main") or soup
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        t = re.sub(r"\s+", " ", el.get_text(" ")).strip()
        if t:
            lines.append(("\n" if el.name.startswith("h") else "") + t)
    (OUT / "cfpb_eft_faq.md").write_text("doc_id: cfpb_eft_faq\nSource: CFPB Electronic Fund Transfers FAQs\n"
                                         + "\n".join(lines) + "\n", encoding="utf-8")

if args.visa_pdf:
    from pypdf import PdfReader
    pages = [pg.extract_text() or "" for pg in PdfReader(args.visa_pdf).pages]
    text = "\n".join(pages)
    # put "Condition 13.1" / "Reason Code 10.4" headings on their own lines for section-aware chunking
    text = re.sub(r"\s*((?:Condition|Reason Code)\s+\d{1,2}\.\d{1,2})\b", r"\n\1", text)
    (OUT / "network_rules.md").write_text("doc_id: network_rules\nSource: Visa Dispute Management Guidelines for Visa Merchants\n"
                                          + text + "\n", encoding="utf-8")

# ------------------------------------------------------------------ check
from dispute_rag.corpus import corpus_section_index, load_documents  # noqa: E402

docs = load_documents(OUT)
idx = corpus_section_index({k: v for k, v in docs.items() if k == "reg_e"})
secs = sorted({r for r in idx if "(" not in r}, key=lambda r: float(r.split(".")[1]) if r.startswith("1005.") else 0)
print(f"\nWrote {', '.join(sorted(p.name for p in OUT.glob('*.md') if p.name.lower() != 'readme.md'))} to {OUT}/")
print(f"reg_e: {len(secs)} sections, {len(idx)} references indexed")
for must in ("1005.2", "1005.6", "1005.6(b)", "1005.6(b)(1)", "1005.11", "1005.11(c)", "1005.11(c)(1)"):
    print(f"  {'OK  ' if must in idx else 'MISSING'} {must}")
if len(secs) < 20 or "1005.11(c)(1)" not in idx:
    print("\nWARNING: fewer references than expected. Open data/corpus/reg_e.md and check that each "
          "'§ 1005.x' heading and each '(a)', '(1)' marker starts its own line.")
print(f"\nRecord in the report: eCFR issue date {issue}.")
