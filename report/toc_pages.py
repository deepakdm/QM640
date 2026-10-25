"""Find the page of each TOC heading in a rendered PDF (pages after the TOC page)."""
import json, re, subprocess, sys
pdf, headings_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
heads = json.load(open(headings_json))
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
pages = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
norm = lambda s: re.sub(r"\s+", " ", s).strip()
start, res = 3, []
for lvl, text in heads:
    found = None
    for i in range(start, n + 1):
        lines = [norm(l) for l in pages[i - 1].splitlines()]
        if any(l == norm(text) for l in lines):
            found = i; break
    if found is None:
        print("not found:", text); found = start
    res.append([lvl, text, found]); start = found
json.dump(res, open(out, "w"))
print(res[:5], "...")
