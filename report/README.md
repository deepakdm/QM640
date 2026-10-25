# Interim report builder

Rebuilds `QM640_Interim_Report.docx` from the analysis outputs. Numbers and figures are taken from
`outputs/eda/` and `outputs/reports/` when they exist; anything still missing is written as a
highlighted [placeholder].

```bash
pip install python-docx
python scripts/01b_eda.py && python scripts/04_compare_and_business.py   # produce the inputs
cd report
python build_interim.py --eda ../outputs/eda --reports ../outputs/reports --out QM640_Interim_Report.docx
```

The table of contents is a Word field. Open the file in Word and press F9 (or right-click the table of
contents and choose Update Field) to refresh its page numbers. `base_style.docx` holds the APA styles:
Times New Roman 12, 1-inch margins and the page-number header.
