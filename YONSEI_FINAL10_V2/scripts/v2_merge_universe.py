#!/usr/bin/env python3
"""Merge per-batch clash-filtered evidence tables into one de-duplicated universe.

Takes any number of EVIDENCE_*.csv tables written by v2_pose_evidence.py and
concatenates them into a single table with one row per (cand_id, pocket).

De-duplication keeps the first table in which a (cand_id, pocket) key appears,
in the order the tables are supplied on the command line. Before dropping a
duplicate the script compares the numeric evidence columns of the duplicate
rows and reports every column that disagrees beyond --tolerance, so a silent
disagreement between two independently produced tables cannot pass unnoticed.

Columns are unioned; a column absent from one input stays missing in the rows
that came from that input. Missing values are never filled with zero.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cand_id", "pocket"]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEY if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {missing}")
    frame["evidence_source"] = path.name
    return frame


def compare_duplicates(frames: list[pd.DataFrame], tolerance: float) -> list[dict]:
    """Report numeric disagreements between duplicate (cand_id, pocket) rows."""
    stacked = pd.concat(frames, ignore_index=True, sort=False)
    dup_keys = stacked[stacked.duplicated(KEY, keep=False)]
    reports: list[dict] = []
    if dup_keys.empty:
        return reports
    numeric = [c for c in stacked.columns
               if c not in KEY and pd.api.types.is_numeric_dtype(stacked[c])]
    for key, group in dup_keys.groupby(KEY, sort=False):
        disagree = {}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values) < 2:
                continue
            spread = float(values.max() - values.min())
            if spread > tolerance:
                disagree[column] = spread
        reports.append(dict(cand_id=key[0], pocket=key[1],
                            copies=int(len(group)),
                            sources=sorted(group.evidence_source.unique()),
                            disagreeing_columns=disagree))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, nargs="+", required=True,
                        help="EVIDENCE_*.csv tables, in precedence order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="numeric spread above which duplicate rows are reported")
    args = parser.parse_args()

    frames = [load(p) for p in args.evidence]
    per_input = {p.name: dict(rows=int(len(f)),
                              unique_cand_ids=int(f.cand_id.nunique()),
                              pairs=int(f.groupby(KEY).ngroups))
                 for p, f in zip(args.evidence, frames)}

    conflicts = compare_duplicates(frames, args.tolerance)

    universe = pd.concat(frames, ignore_index=True, sort=False)
    before = len(universe)
    universe = universe.drop_duplicates(KEY, keep="first").reset_index(drop=True)
    dropped = before - len(universe)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(args.out, index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        inputs=[str(p) for p in args.evidence],
        per_input=per_input,
        rows_before_dedup=int(before),
        rows_after_dedup=int(len(universe)),
        duplicate_rows_dropped=int(dropped),
        duplicate_keys=len(conflicts),
        duplicate_keys_with_numeric_disagreement=sum(
            1 for c in conflicts if c["disagreeing_columns"]),
        duplicate_detail=conflicts[:200],
        unique_cand_ids=int(universe.cand_id.nunique()),
        pairs_by_pocket={k: int(v) for k, v in universe.pocket.value_counts().items()},
        cand_ids_by_pocket={k: int(sub.cand_id.nunique())
                            for k, sub in universe.groupby("pocket")},
        rows_by_source={k: int(v) for k, v in universe.evidence_source.value_counts().items()},
        molecules_with_no_acceptable_pose=(
            int(universe.all_poses_clashing.sum())
            if "all_poses_clashing" in universe else None),
        note=("One row per ligand per pocket. Docking and fingerprint evidence only; "
              "no activity claim. Missing values are missing, not zero."),
    )
    out_summary = args.summary or args.out.with_suffix(".summary.json")
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    printable = dict(summary)
    printable["duplicate_detail"] = f"{len(conflicts)} keys (see {out_summary})"
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
