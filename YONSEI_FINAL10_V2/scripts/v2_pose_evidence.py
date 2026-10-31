#!/usr/bin/env python3
"""Per-molecule evidence built from CLASH-FILTERED poses.

PIGNet2 rescoring of the EX8 pose set showed that a large share of GNINA poses
are sterically clashing, and GNINA's own minimizedAffinity independently agrees
(clashing poses score strongly positive). A clashing pose can accumulate many
contacts and therefore a spuriously high interaction-fingerprint recovery, so
the best-IFP pose must be chosen among physically plausible poses only.

A pose is accepted when its GNINA minimizedAffinity is negative. Molecules with
no acceptable pose are kept and flagged rather than dropped.
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
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--pignet", type=Path, default=None)
    parser.add_argument("--ligand-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", default="batch1")
    parser.add_argument("--max-affinity", type=float, default=0.0,
                        help="a pose is acceptable when minimizedAffinity is below this")
    args = parser.parse_args()

    poses = pd.read_csv(args.poses)
    ifp = pd.read_csv(args.interactions)
    lig = pd.read_csv(args.ligand_table)
    keep = [c for c in ["cand_id", "pocket", "pose_rank", "ifp_recovery",
                        "ifp_n_interactions", "ifp_hbonds", "ifp_keys"] if c in ifp]
    ifp = ifp[keep]

    p = poses.merge(ifp, on=["cand_id", "pocket", "pose_rank"], how="left")
    if args.pignet and args.pignet.exists():
        pig = pd.read_csv(args.pignet)
        cols = {c.lower(): c for c in pig.columns}
        idx = cols.get("pose_index") or cols.get("pose_rank")
        ens = next((c for c in pig.columns if "ensemble" in c.lower() and "mean" in c.lower()), None)
        if idx and ens:
            pig = pig.rename(columns={idx: "pose_rank", ens: "pignet2"})
            pig["pignet2"] = pd.to_numeric(pig.pignet2, errors="coerce")
            p = p.merge(pig[["cand_id", "pocket", "pose_rank", "pignet2"]],
                        on=["cand_id", "pocket", "pose_rank"], how="left")
    if "pignet2" not in p:
        p["pignet2"] = np.nan

    p["pose_acceptable"] = pd.to_numeric(p.minimizedAffinity, errors="coerce") < args.max_affinity
    n_total, n_ok = len(p), int(p.pose_acceptable.sum())

    rows = []
    for (cand, pocket), g in p.groupby(["cand_id", "pocket"], sort=False):
        ok = g[g.pose_acceptable]
        pool = ok if len(ok) else g
        top = g.sort_values("pose_rank").iloc[0]
        by_aff = pool.sort_values("minimizedAffinity", na_position="last").iloc[0]
        by_ifp = pool.sort_values("ifp_recovery", ascending=False, na_position="last").iloc[0]
        by_pig = (pool.sort_values("pignet2", na_position="last").iloc[0]
                  if pool.pignet2.notna().any() else by_aff)
        heavy = top.get("heavy_atoms", np.nan)
        rows.append(dict(
            batch=args.batch, cand_id=cand, pocket=pocket,
            n_poses=len(g), n_poses_acceptable=len(ok),
            all_poses_clashing=bool(len(ok) == 0),
            cnn_ranked_pose=1,
            cnnpose_minimizedAffinity=top.minimizedAffinity, cnnpose_CNNscore=top.CNNscore,
            cnnpose_CNNaffinity=top.CNNaffinity, cnnpose_CNN_VS=top.CNN_VS,
            best_vina_pose=int(by_aff.pose_rank), best_vina_affinity=by_aff.minimizedAffinity,
            best_CNNscore=pool.CNNscore.max(), best_CNNaffinity=pool.CNNaffinity.max(),
            best_CNN_VS=pool.CNN_VS.max(),
            max_ifp_recovery=by_ifp.ifp_recovery, pose_of_max_ifp=int(by_ifp.pose_rank),
            ifp_n_interactions=by_ifp.ifp_n_interactions, ifp_hbonds=by_ifp.ifp_hbonds,
            ifp_keys=by_ifp.get("ifp_keys", ""),
            atbestifp_minimizedAffinity=by_ifp.minimizedAffinity,
            atbestifp_CNNscore=by_ifp.CNNscore,
            pignet2_best=by_pig.pignet2, pignet2_best_pose=int(by_pig.pose_rank),
            pignet2_at_best_vina=by_aff.pignet2, pignet2_at_best_ifp=by_ifp.pignet2,
            heavy_atoms=heavy,
            ligand_efficiency=(-by_aff.minimizedAffinity / heavy)
            if heavy and not pd.isna(heavy) and not pd.isna(by_aff.minimizedAffinity) else np.nan,
        ))
    e = pd.DataFrame(rows).merge(lig, on="cand_id", how="left", suffixes=("", "_lig"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    e.to_csv(args.out, index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), batch=args.batch,
        poses_total=n_total, poses_acceptable=n_ok,
        poses_rejected_as_clashing=n_total - n_ok,
        clash_rejection_rate=round(1 - n_ok / n_total, 4),
        molecules=int(len(e)),
        molecules_with_no_acceptable_pose=int(e.all_poses_clashing.sum()),
        pignet2_available=bool(e.pignet2_best.notna().any()),
        ifp_recovery_by_pocket={
            p_: dict(median=round(float(s.max_ifp_recovery.median()), 4),
                     max=round(float(s.max_ifp_recovery.max()), 4),
                     with_hbond=int((s.ifp_hbonds > 0).sum()))
            for p_, s in e.groupby("pocket")},
        note=("best-Vina, best-IFP and best-PIGNet2 poses are chosen among "
              "non-clashing poses only; molecules with no acceptable pose are "
              "flagged, not dropped."),
    )
    (args.out.parent / f"POSE_EVIDENCE_SUMMARY_{args.batch}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
