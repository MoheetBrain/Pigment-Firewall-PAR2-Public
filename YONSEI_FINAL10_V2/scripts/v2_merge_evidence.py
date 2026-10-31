#!/usr/bin/env python3
"""Assemble the per-molecule evidence table for a de novo screen batch.

One row per ligand per pocket, carrying the docking evidence, the repaired
reference-antagonist interaction fingerprint evidence, and the RDKit properties,
with every metric ranked WITHIN its pocket. 5NDD and 5NDZ are never pooled on
raw scores.

Three distinct poses are reported per row because they are usually different
molecules' worth of information:
  * CNN-ranked pose  - always pose 1, because --cnn_scoring rescore makes GNINA
    order its output by CNNscore
  * best-Vina pose   - the pose with the lowest minimizedAffinity
  * best-IFP pose    - the pose recovering most of the crystal fingerprint
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--ligand-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", default="batch1")
    args = parser.parse_args()

    poses = pd.read_csv(args.screen / "V2_POSE_SCORES.csv")
    agg = pd.read_csv(args.screen / "V2_LIGAND_POCKET_SCORES.csv")
    lig = pd.read_csv(args.ligand_table)
    ifp_path = args.screen / "V2_POSE_INTERACTIONS_SUMMARY.csv"
    if not ifp_path.exists():
        raise SystemExit(f"missing {ifp_path}; run v2_interactions.py first")
    ifp = pd.read_csv(ifp_path)

    # Scores at the CNN-ranked pose (pose 1) and at the best-IFP pose.
    p1 = (poses[poses.pose_rank == 1]
          .set_index(["cand_id", "pocket"])[["minimizedAffinity", "CNNscore",
                                             "CNNaffinity", "CNN_VS"]]
          .rename(columns=lambda c: f"cnnpose_{c}"))
    key = ifp[["cand_id", "pocket", "ifp_best_pose_rank"]].rename(
        columns={"ifp_best_pose_rank": "pose_rank"})
    at_ifp = (poses.merge(key, on=["cand_id", "pocket", "pose_rank"])
              .set_index(["cand_id", "pocket"])[["minimizedAffinity", "CNNscore",
                                                 "CNNaffinity", "CNN_VS"]]
              .rename(columns=lambda c: f"atbestifp_{c}"))

    e = (agg.merge(lig, on="cand_id", how="left", suffixes=("", "_lig"))
            .merge(ifp.drop(columns=["ifp_keys"], errors="ignore"),
                   on=["cand_id", "pocket"], how="left")
            .merge(p1, on=["cand_id", "pocket"], how="left")
            .merge(at_ifp, on=["cand_id", "pocket"], how="left"))
    e["batch"] = args.batch
    e = e.rename(columns={"ifp_recovery": "max_ifp_recovery",
                          "ifp_best_pose_rank": "pose_of_max_ifp",
                          "best_affinity_pose_rank": "best_vina_pose",
                          "best_affinity": "best_vina_affinity"})
    e["cnn_ranked_pose"] = 1

    # Within-pocket ranking on each metric separately. No cross-pocket pooling.
    for metric, higher_better in [("best_vina_affinity", False), ("best_CNNscore", True),
                                  ("best_CNNaffinity", True), ("best_CNN_VS", True),
                                  ("ligand_efficiency", True), ("max_ifp_recovery", True)]:
        e[f"rank_in_pocket_{metric}"] = (
            e.groupby("pocket")[metric]
             .rank(ascending=not higher_better, method="min").astype("Int64"))
        e[f"pct_in_pocket_{metric}"] = (
            e.groupby("pocket")[metric].rank(pct=True, ascending=higher_better).round(4))

    cols = [
        "batch", "cand_id", "library_rank", "denovo_family", "denovo_rank", "design_score",
        "pocket", "tautomer_corrected", "smiles_prepared", "inchikey", "murcko",
        "max_ifp_recovery", "pose_of_max_ifp", "ifp_n_interactions", "ifp_hbonds",
        "ifp_shared_keys", "ifp_reference_keys",
        "cnn_ranked_pose", "cnnpose_minimizedAffinity", "cnnpose_CNNscore",
        "cnnpose_CNNaffinity", "cnnpose_CNN_VS",
        "best_vina_pose", "best_vina_affinity",
        "atbestifp_minimizedAffinity", "atbestifp_CNNscore", "atbestifp_CNNaffinity",
        "best_CNNscore", "best_CNNaffinity", "best_CNN_VS",
        "ligand_efficiency", "heavy_atoms", "n_poses",
        "mw", "clogp", "tpsa", "hba", "hbd", "rotb", "fsp3", "qed", "formal_charge",
    ] + [c for c in e.columns if c.startswith(("rank_in_pocket_", "pct_in_pocket_"))]
    cols = [c for c in cols if c in e.columns]
    table = e[cols].sort_values(["pocket", "rank_in_pocket_max_ifp_recovery"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), batch=args.batch,
        rows=int(len(table)), unique_ligands=int(table.cand_id.nunique()),
        by_pocket={k: int(v) for k, v in table.pocket.value_counts().items()},
        by_family={k: int(v) for k, v in table.drop_duplicates("cand_id")
                   .denovo_family.value_counts().items()},
        ifp_recovery_by_pocket={
            p: dict(median=round(float(s.max_ifp_recovery.median()), 4),
                    p90=round(float(s.max_ifp_recovery.quantile(.9)), 4),
                    max=round(float(s.max_ifp_recovery.max()), 4),
                    ligands_with_hbond=int((s.ifp_hbonds > 0).sum()))
            for p, s in table.groupby("pocket")},
        pose_disagreement=dict(
            cnn_pose_is_best_vina=int((table.best_vina_pose == 1).sum()),
            cnn_pose_is_best_ifp=int((table.pose_of_max_ifp == 1).sum()),
            best_vina_equals_best_ifp=int((table.best_vina_pose == table.pose_of_max_ifp).sum()),
            total=int(len(table))),
        note=("Ranks are computed within each pocket. 5NDD and 5NDZ are never pooled "
              "on raw scores. IFP recovery is geometric similarity to the repaired "
              "crystal reference, not evidence of activity."),
    )
    (args.out.parent / f"EVIDENCE_SUMMARY_{args.batch}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
