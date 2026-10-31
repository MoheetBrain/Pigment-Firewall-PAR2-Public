# Inspecting and reproducing the study

## Verify saved evidence

Run `python3 scripts/verify_release.py` to check the publication file hashes.
The two original panel CSVs are byte-identical to their archived originals.
The latest panel includes exact structures and all recorded metrics. Its
`PANEL_PROVENANCE.json` has one path made repository-relative; numbers are unchanged.
Small original evidence inputs (80 molecules, 160 ligand-pocket rows and 32
frozen seeds) are included to audit the panel-building stage. They are research
results, not a redistributable chemical catalogue.

## Local software and inputs

Use Python 3.11 or a compatible scientific environment with RDKit, NumPy,
pandas, SciPy, requests, ProLIF, MDAnalysis, Meeko and the dependencies required
by the selected script. The campaign used GNINA v1.3.3 and Boltz 2.2.1; neither
binary nor model weights is redistributed. Exact fully locked environments for
all original lanes were not saved; complete environment-level reproduction has
not been demonstrated by this release.

The small saved inputs support an offline check of the final table-building
stage. With RDKit, pandas and NumPy installed, run from the repository root:

```bash
python YONSEI_FINAL10_V2/scripts/v3_final_sourceable_panel.py \
  --ex32 YONSEI_FINAL10_V2/11_SOURCEABLE/analogues/REAL_ANALOGUES_EX32.csv \
  --analogues YONSEI_FINAL10_V2/11_SOURCEABLE/analogues/EX32_FINALIST_QUEUE_WITH_VENDOR.csv \
  --sourceability YONSEI_FINAL10_V2/11_SOURCEABLE/analogues/FINALIST_VENDOR_CHECK.csv \
  --seeds YONSEI_FINAL10_V2/11_SOURCEABLE/seeds/FINAL_DE_NOVO_SIMILARITY_SEEDS.csv \
  --out reproduced-panel
```

Compare the result with the unchanged archived CSV; do not overwrite the archive.
This checks selection from saved evidence, not the correctness of docking or the
missing functional gates. To repeat upstream computation, obtain CC0 5NDD/5NDZ
coordinates and molecular libraries from their lawful sources under their terms.
Bulk libraries, full prepared poses and model weights are not included, so complete
screen-level reproduction remains limited by input availability and environment
capture.

Portable corrected preparation code is
`YONSEI_FINAL10_V2/scripts/v3_prepare_finalists.py`; panel selection is
`v3_final_sourceable_panel.py`. Use `--help` after installing dependencies and pass
explicit local input/output paths. Run from this repository root, using a separate
output directory. Retained v2 scripts document historical defects; use the v3
finalist preparation for that stage. No remote-login scripts are included.

Publication validation checked Python syntax, CSV/hash consistency and the offline
selection reconstruction described in [VALIDATION.md](VALIDATION.md). No full
docking, model inference, library search or wet-lab validation was rerun.
