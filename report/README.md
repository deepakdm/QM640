# Report builders

Both builders read the analysis outputs and write an APA-formatted Word document from
`base_style.docx` (Times New Roman 12, 1-inch margins, page-number header). Anything not yet
available is written as a highlighted [placeholder]; nothing is invented.

## Final report (draft mode)

```bash
cd report
python build_final.py                  # uses the test set if outputs exist for it, else the development set
python build_final.py --primary dev    # force development-set numbers
python check_final.py QM640_Final_Report_DRAFT.docx
```

`build_final.py` follows the final-report rubric order and fills every table, figure and number from
`outputs/final/`, `outputs/reports*/`, `outputs/eda*/` and `data/splits/`. Author prose is left as
placeholders naming the rubric criterion. `check_final.py` checks the abstract length and paragraph
count, remaining placeholders, table and figure numbering and call-outs, abbreviations at first use,
reference coverage and page setup.

## Interim report (submitted 27 October 2026)

```bash
python build_interim.py --eda ../outputs/eda --reports ../outputs/reports --out QM640_Interim_Report.docx
```

The table of contents is a Word field: open the file in Word and press F9 (or right-click it and
choose Update Field) to refresh page numbers. `toc_pages.py` reads page numbers from a PDF render
for the static table of contents.
