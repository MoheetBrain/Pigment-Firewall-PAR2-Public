#!/usr/bin/env python3
"""Collect GNINA pose scores from a V2 screen stage.

Reads the gzipped output SDFs written by v2_run_screen.sh, extracts every pose
score, and writes a per-pose table plus a per-ligand-per-pocket aggregate.
Optionally emits refine-stage shards for the best ligands.

No ranking decision is made here beyond ordering; interpretation belongs to
v2_rank_and_select.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SCORE_FIELDS = [
    "minimizedAffinity",
    "CNNscore",
    "CNNaffinity",
    "CNN_VS",
    "CNNaffinity_variance",
    "minimizedRMSD",
]


def read_pose_file(path: Path, pocket: str) -> list[dict]:
    rows: list[dict] = []
    pose_index: dict[str, int] = defaultdict(int)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            cand_id = mol.GetProp("cand_id") if mol.HasProp("cand_id") else name
            if not cand_id:
                continue
            pose_index[cand_id] += 1
            row = {
                "cand_id": cand_id,
                "pocket": pocket,
                "pose_rank": pose_index[cand_id],
                "shard": path.name,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            }
            for field in SCORE_FIELDS:
                value = None
                if mol.HasProp(field):
                    try:
                        value = float(mol.GetProp(field))
                    except ValueError:
                        value = None
                row[field] = value
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True, help="stage output dir (contains poses/)")
    parser.add_argument("--ligand-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-refine", type=int, default=0,
                        help="write refine shards for the best N ligands per pocket")
    parser.add_argument("--refine-shard-size", type=int, default=100)
    parser.add_argument("--prepared-shards", type=Path, default=None,
                        help="original shard dir, required with --emit-refine")
    args = parser.parse_args()

    screen = args.screen.resolve()
    out = (args.out or screen).resolve()
    out.mkdir(parents=True, exist_ok=True)
    poses_dir = screen / "poses"
    pose_files = sorted(p for p in poses_dir.glob("*_out.sdf*") if not p.name.startswith("."))
    if not pose_files:
        raise SystemExit(f"no pose files under {poses_dir}")

    rows: list[dict] = []
    for path in pose_files:
        pocket = path.name.split("_shard")[0]
        rows.extend(read_pose_file(path, pocket))
    poses = pd.DataFrame(rows)
    if poses.empty:
        raise SystemExit("no poses parsed")
    poses.to_csv(out / "V2_POSE_SCORES.csv", index=False)

    ligands = pd.read_csv(args.ligand_table)
    heavy = dict(zip(ligands.cand_id, ligands.heavy_atoms))

    agg_rows = []
    for (cand_id, pocket), group in poses.groupby(["cand_id", "pocket"], sort=False):
        group = group.sort_values("pose_rank")
        top = group.iloc[0]
        by_aff = group.sort_values("minimizedAffinity", na_position="last").iloc[0]
        by_cnnvs = group.sort_values("CNN_VS", ascending=False, na_position="last").iloc[0]
        n_heavy = heavy.get(cand_id, top["heavy_atoms"]) or top["heavy_atoms"]
        best_aff = by_aff["minimizedAffinity"]
        agg_rows.append(
            dict(
                cand_id=cand_id,
                pocket=pocket,
                n_poses=len(group),
                toppose_affinity=top["minimizedAffinity"],
                toppose_CNNscore=top["CNNscore"],
                toppose_CNNaffinity=top["CNNaffinity"],
                toppose_CNN_VS=top["CNN_VS"],
                best_affinity=best_aff,
                best_affinity_pose_rank=int(by_aff["pose_rank"]),
                best_CNN_VS=by_cnnvs["CNN_VS"],
                best_CNN_VS_pose_rank=int(by_cnnvs["pose_rank"]),
                best_CNNscore=group["CNNscore"].max(),
                best_CNNaffinity=group["CNNaffinity"].max(),
                heavy_atoms=n_heavy,
                ligand_efficiency=(-best_aff / n_heavy) if (best_aff is not None and not
                                                            (isinstance(best_aff, float) and math.isnan(best_aff))
                                                            and n_heavy) else None,
            )
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "V2_LIGAND_POCKET_SCORES.csv", index=False)

    summary = dict(
        collected_utc=datetime.now(timezone.utc).isoformat(),
        pose_files=len(pose_files),
        poses_parsed=int(len(poses)),
        ligand_pocket_pairs=int(len(agg)),
        unique_ligands_scored=int(agg.cand_id.nunique()),
        pairs_by_pocket={k: int(v) for k, v in agg.pocket.value_counts().items()},
        affinity_summary={
            pocket: dict(
                min=float(sub.best_affinity.min()),
                median=float(sub.best_affinity.median()),
                max=float(sub.best_affinity.max()),
            )
            for pocket, sub in agg.groupby("pocket")
            if sub.best_affinity.notna().any()
        },
        note="Docking scores only. Not a measure of biological activity.",
    )

    if args.emit_refine:
        if args.prepared_shards is None:
            raise SystemExit("--emit-refine requires --prepared-shards")
        keep: dict[str, set[str]] = {}
        for pocket, sub in agg.groupby("pocket"):
            ranked = sub.sort_values(["best_CNN_VS", "best_affinity"], ascending=[False, True])
            keep[pocket] = set(ranked.head(args.emit_refine).cand_id)
        refine_dir = out / "refine_shards"
        refine_dir.mkdir(exist_ok=True)
        written = defaultdict(int)
        buffers: dict[str, list[str]] = defaultdict(list)
        counters: dict[str, int] = defaultdict(int)

        def flush(pocket: str) -> None:
            if not buffers[pocket]:
                return
            path = refine_dir / f"{pocket}_shard{counters[pocket]:04d}.sdf"
            path.write_text("".join(buffers[pocket]))
            buffers[pocket] = []
            counters[pocket] += 1

        for shard in sorted(args.prepared_shards.glob("*.sdf")):
            pocket = shard.name.split("_shard")[0]
            if pocket not in keep:
                continue
            text = shard.read_text()
            for record in text.split("$$$$\n"):
                if not record.strip():
                    continue
                name = record.lstrip("\n").split("\n", 1)[0].strip()
                if name in keep[pocket]:
                    buffers[pocket].append(record + "$$$$\n")
                    written[pocket] += 1
                    if len(buffers[pocket]) >= args.refine_shard_size:
                        flush(pocket)
        for pocket in list(buffers):
            flush(pocket)
        summary["refine"] = dict(
            requested_per_pocket=args.emit_refine,
            written_by_pocket={k: int(v) for k, v in written.items()},
            shard_dir=str(refine_dir),
            selection_rule="top N per pocket by best_CNN_VS then best_affinity",
        )

    (out / "V2_COLLECT_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
