Place corpus files here as .md/.txt. First line must be `doc_id: <id>`.

Keep headings intact so section-aware chunking can attach clause references:
  * CFR sections:    "§ 1005.11 Procedures for resolving errors."   (Reg E)
                     "§ 1026.13 Billing error resolution."          (Reg Z)
  * Sub-paragraphs:  "(a)", "(b)(1)"
  * Network guides:  "Condition 13.1 ..." / "Reason Code 10.4 ..."  (Visa merchant guide)

Documents without recognizable headings fall back to fixed-size chunks with no section
reference, which lowers their retrieval-recall score. See docs/data_sources.md for links.
