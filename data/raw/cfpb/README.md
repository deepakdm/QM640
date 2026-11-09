# Raw CFPB data

Place the merged CFPB narratives file here as **`complaints.csv`**. It is git-ignored and no script
downloads it unless asked. Requirements, source and the list of steps that need it are in
`docs/data_sources.md`.

- `complaints.csv`: expected local file (you provide it).
- `complaints.csv.gz`: the same data committed through Git LFS. A plain clone holds only a small
  placeholder; run `git lfs pull` to fetch the real file, which the scripts then use as a fallback.
- `archive/`: optional folder for the CFPB archive ZIPs read by `scripts/00a_build_from_archive.py`.

Check the file with `python scripts/00_check_export.py`.
