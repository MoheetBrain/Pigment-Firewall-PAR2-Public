#!/usr/bin/env python3
"""Select a DIVERSE ~500-molecule set for EX32 precision redocking.

This is deliberately NOT "top 500 by composite score". A composite ranking
collapses every axis of evidence onto one number and then returns whichever
chemotype happens to win that particular weighting. The EX32 budget is spent
instead on a set that spans the independent axes on which a design could turn
out to be right, so that a later failure of one axis does not take the whole
refined set with it.

Axes (quota-allocated, deduplicated, scaffold-capped):

  ifp_recovery     best recovery of the repaired crystal fingerprint
  dual_anchor      recapitulates BOTH repaired crystal hydrogen bonds
                   (TYR:82 and HIS:135 in 5NDD -- see note below)
  anchor_protomer  protomer-robust anchor: any two of TYR:82 / HIS:135 / ASP:228
  design_score     best de novo design-hotspot score
  physics          best GNINA Vina affinity (and PIGNet2 where available)
  ligand_eff       compact, high ligand-efficiency chemistry
  az8838_adjacent  highest Tanimoto to the crystallographic antagonist
  fsp3             high-Fsp3 three-dimensional chemistry
  envelope         inside the favourable topical property envelope
  cluster          representatives of convergent (many-member) scaffolds
  scaffold_hop     rare / singleton scaffolds far from the reference
  stereo_redock    stereoisomers corrected upstream, if a list is supplied

ANCHOR DEFINITION. The repaired crystal fingerprint
(02_REFERENCE_VALIDATION_REPAIRED/REFERENCE_IFP_REPAIRED.json, condition D)
contains exactly two hydrogen bonds in the 5NDD orthosteric pocket:
TYR:82:A|HBDonor and HIS:135:A|HBDonor. ASP:228 appears only as a VdWContact.
The "dual anchor" here is therefore TYR:82 + HIS:135, read from that file at
run time rather than hard-coded, superseding the ASP:228+HIS:135 pair used by
v2_consensus_rank.py. The 5NDZ allosteric reference contains no hydrogen bond
at all, so the dual-anchor axis is not defined there and is not used.

Because the ASP:228 / HIS:135 protonation state is genuinely uncertain, a second
"any two of TYR:82 / HIS:135 / ASP:228" axis is carried as a protomer-robust
hedge. Both definitions reproduce 10_FINAL/ANCHOR_DEFINITION_AUDIT.csv exactly
(81 and 171 designs respectively on the 9,066 rows that audit covers).

Missing values stay missing. Nothing is imputed and nothing is scored as zero.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HBOND_TYPES = ("HBDonor", "HBAcceptor")

# Favourable topical property envelope (given; not fitted here).
ENVELOPE = dict(mw=(350.0, 480.0), clogp=(1.5, 3.0), tpsa=(70.0, 95.0),
                hbd_max=2, aromatic_rings_max=2, fsp3_min=0.35, rotb_max=5)


def repaired_hbond_anchors(root: Path) -> dict[str, list[str]]:
    """Residues that hydrogen-bond to the ligand in the repaired crystal pose."""
    path = root / "YONSEI_FINAL10_V2" / "02_REFERENCE_VALIDATION_REPAIRED" / "REFERENCE_IFP_REPAIRED.json"
    data = json.loads(path.read_text())
    anchors: dict[str, list[str]] = {}
    for spec in data["systems"].values():
        residues = sorted({
            key.split("|")[0] for key in spec["reference_fingerprint_keys"]
            if key.split("|")[1] in HBOND_TYPES
        })
        anchors[spec["pocket"]] = residues
    return anchors


def add_derived(df: pd.DataFrame, anchors: dict[str, list[str]]) -> pd.DataFrame:
    """Anchor recapitulation and the ring counts the envelope needs."""
    df = df.copy()

    def anchor_hbonds(row) -> float:
        wanted = anchors.get(row.pocket, [])
        if not wanted:
            return np.nan          # undefined for this pocket -- stays missing
        keys = str(row.ifp_keys).split(";")
        hit = {k.split("|")[0] for k in keys
               if "|" in k and k.split("|")[1] in HBOND_TYPES}
        return sum(res in hit for res in wanted) / len(wanted)

    df["anchor_hbond_fraction_repaired"] = df.apply(anchor_hbonds, axis=1)
    df["dual_anchor_repaired"] = df.anchor_hbond_fraction_repaired >= 0.999

    # Protomer-robust hedge: any two of the three candidate polar positions.
    protomer_set = ["TYR:82:A", "HIS:135:A", "ASP:228:A"]

    def any_two(row) -> float:
        if not anchors.get(row.pocket):
            return np.nan
        keys = str(row.ifp_keys).split(";")
        hit = {k.split("|")[0] for k in keys
               if "|" in k and k.split("|")[1] in HBOND_TYPES}
        return float(sum(res in hit for res in protomer_set) >= 2)

    df["anchor_any_two_repaired"] = df.apply(any_two, axis=1).astype("boolean")

    rings, arom = [], []
    for smi in df.smiles_prepared:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rings.append(np.nan); arom.append(np.nan); continue
        rings.append(mol.GetRingInfo().NumRings())
        arom.append(sum(1 for r in mol.GetRingInfo().AtomRings()
                        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)))
    df["n_rings"] = rings
    df["aromatic_rings"] = arom

    e = ENVELOPE
    df["in_envelope"] = (
        df.mw.between(*e["mw"]) & df.clogp.between(*e["clogp"])
        & df.tpsa.between(*e["tpsa"]) & (df.hbd <= e["hbd_max"])
        & (df.aromatic_rings <= e["aromatic_rings_max"])
        & (df.fsp3 >= e["fsp3_min"]) & (df.rotb <= e["rotb_max"]))

    # Distance from the centre of the envelope, for ranking inside it.
    def mid(bounds):
        lo, hi = bounds
        return (lo + hi) / 2.0, (hi - lo) / 2.0
    score = 0.0
    for col in ("mw", "clogp", "tpsa"):
        centre, half = mid(e[col])
        score = score + ((df[col] - centre) / half).abs()
    df["envelope_distance"] = score
    return df


def take(pool: pd.DataFrame, order_col: str, ascending: bool, quota: int,
         chosen: set, scaffold_used: Counter, scaffold_cap: int,
         axis: str, extra_mask=None) -> list[tuple]:
    """Greedy top-of-axis pick with a per-scaffold cap and global dedup."""
    sub = pool if extra_mask is None else pool[extra_mask]
    sub = sub.dropna(subset=[order_col]).sort_values(order_col, ascending=ascending)
    picked = []
    for row in sub.itertuples():
        if len(picked) >= quota:
            break
        key = (row.cand_id, row.pocket)
        if key in chosen:
            continue
        scaffold = row.murcko if isinstance(row.murcko, str) and row.murcko else f"__none__{row.cand_id}"
        if scaffold_used[scaffold] >= scaffold_cap:
            continue
        chosen.add(key)
        scaffold_used[scaffold] += 1
        picked.append((key, axis))
    return picked


def select_pocket(df: pd.DataFrame, pocket: str, quotas: dict[str, int],
                  scaffold_cap: int, stereo_ids: set[str]) -> pd.DataFrame:
    pool = df[df.pocket == pocket]
    chosen: set = set()
    scaffold_used: Counter = Counter()
    assignment: dict[tuple, str] = {}

    def record(picks):
        for key, axis in picks:
            assignment.setdefault(key, axis)

    # Convergent clusters: scaffolds with many independent members.
    counts = pool.murcko.value_counts()
    convergent = set(counts[counts >= 8].index)
    singleton = set(counts[counts == 1].index)

    plan = [
        ("stereo_redock", "max_ifp_recovery", False,
         pool.cand_id.isin(stereo_ids) if stereo_ids else None),
        ("ifp_recovery", "max_ifp_recovery", False, None),
        ("dual_anchor", "best_vina_affinity", True,
         pool.dual_anchor_repaired.fillna(False)),
        ("anchor_protomer", "best_vina_affinity", True,
         pool.anchor_any_two_repaired.fillna(False).astype(bool)),
        ("design_score", "design_score", False, None),
        ("physics", "best_vina_affinity", True, None),
        ("ligand_eff", "ligand_efficiency", False, None),
        ("az8838_adjacent", "ref_tanimoto", False, None),
        ("fsp3", "fsp3", False, None),
        ("envelope", "envelope_distance", True, pool.in_envelope.fillna(False)),
        ("cluster", "best_vina_affinity", True, pool.murcko.isin(convergent)),
        ("scaffold_hop", "best_vina_affinity", True,
         pool.murcko.isin(singleton) & (pool.ref_tanimoto < pool.ref_tanimoto.median())),
    ]
    for axis, col, asc, mask in plan:
        quota = quotas.get(axis, 0)
        if quota <= 0 or (mask is not None and not mask.any()):
            continue
        record(take(pool, col, asc, quota, chosen, scaffold_used,
                    scaffold_cap, axis, mask))

    out = pool[[(c, p) in chosen for c, p in zip(pool.cand_id, pool.pocket)]].copy()
    out["ex32_axis"] = [assignment[(c, p)] for c, p in zip(out.cand_id, out.pocket)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--stereo-list", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-5ndd", type=int, default=400)
    ap.add_argument("--target-5ndz", type=int, default=100)
    ap.add_argument("--scaffold-cap", type=int, default=4)
    args = ap.parse_args()

    universe = pd.read_csv(args.universe, low_memory=False)
    anchors = repaired_hbond_anchors(args.root)
    print("repaired hydrogen-bond anchors:", anchors)

    # Only molecules with usable pose geometry are eligible for refinement.
    eligible = universe[~universe.all_poses_clashing.astype(bool)].copy()
    print(f"universe rows {len(universe)}, eligible after clash filter {len(eligible)}")
    eligible = add_derived(eligible, anchors)

    stereo_ids: set[str] = set()
    if args.stereo_list and args.stereo_list.exists():
        s = pd.read_csv(args.stereo_list)
        col = "cand_id" if "cand_id" in s.columns else s.columns[0]
        stereo_ids = set(s[col].astype(str))
        print(f"stereo redock list: {len(stereo_ids)} ids from {args.stereo_list.name}")
    else:
        print("stereo redock list: NOT AVAILABLE (axis skipped)")

    q5ndd = dict(stereo_redock=20, ifp_recovery=55, dual_anchor=40,
                 anchor_protomer=25, design_score=45, physics=42, ligand_eff=38,
                 az8838_adjacent=33, fsp3=33, envelope=42, cluster=27,
                 scaffold_hop=20)
    q5ndz = dict(stereo_redock=5, ifp_recovery=25, dual_anchor=0,
                 anchor_protomer=0, design_score=15, physics=20, ligand_eff=10,
                 az8838_adjacent=5, fsp3=10, envelope=15, cluster=5,
                 scaffold_hop=5)
    if not stereo_ids:
        q5ndd["stereo_redock"] = 0
        q5ndz["stereo_redock"] = 0

    parts = []
    for pocket, quotas, target in (("5NDD_orthosteric", q5ndd, args.target_5ndd),
                                   ("5NDZ_allosteric", q5ndz, args.target_5ndz)):
        sel = select_pocket(eligible, pocket, quotas, args.scaffold_cap, stereo_ids)
        # Top-up on IFP recovery if quotas under-filled (rare axes can starve).
        if len(sel) < target:
            pool = eligible[(eligible.pocket == pocket)
                            & ~eligible.cand_id.isin(sel.cand_id)]
            extra = pool.sort_values("max_ifp_recovery", ascending=False).head(target - len(sel)).copy()
            extra["ex32_axis"] = "topup_ifp_recovery"
            sel = pd.concat([sel, extra], ignore_index=True)
        parts.append(sel)
        print(f"{pocket}: selected {len(sel)}")

    selection = pd.concat(parts, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.out_dir / "EX32_SELECTION.csv", index=False)
    selection[["cand_id"]].drop_duplicates().to_csv(args.out_dir / "EX32_CAND_IDS.csv", index=False)
    # Carve lists, split by which prepared library holds the 3D coordinates.
    # batch1 ids were prepared into library/; batch2, recovered and gapfill ids
    # were prepared into library_batch2/. Nothing is re-embedded downstream.
    lib_of = {"batch1": "library"}
    selection["prep_library"] = selection.batch.map(lambda b: lib_of.get(b, "library_batch2"))
    carve_counts = {}
    for pocket, short in (("5NDD_orthosteric", "5NDD"), ("5NDZ_allosteric", "5NDZ")):
        per_pocket = selection[selection.pocket == pocket]
        per_pocket[["cand_id"]].drop_duplicates().to_csv(
            args.out_dir / f"EX32_CAND_IDS_{short}.csv", index=False)
        for lib in ("library", "library_batch2"):
            ids = per_pocket[per_pocket.prep_library == lib][["cand_id"]].drop_duplicates()
            path = args.out_dir / f"EX32_IDS_{lib}_{short}.csv"
            ids.to_csv(path, index=False)
            carve_counts[path.name] = int(len(ids))
    print("carve lists:", json.dumps(carve_counts, indent=2))

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        universe=str(args.universe),
        universe_rows=int(len(universe)),
        eligible_rows=int(len(eligible)),
        clash_filtered_out=int(len(universe) - len(eligible)),
        anchor_definition={k: v for k, v in anchors.items()},
        anchor_note=("TYR:82 + HIS:135 read from the repaired crystal fingerprint; "
                     "supersedes ASP:228 + HIS:135 in v2_consensus_rank.py"),
        stereo_list_used=bool(stereo_ids),
        selected_jobs=int(len(selection)),
        selected_molecules=int(selection.cand_id.nunique()),
        by_pocket=selection.pocket.value_counts().to_dict(),
        by_axis=selection.ex32_axis.value_counts().to_dict(),
        unique_murcko_scaffolds=int(selection.murcko.nunique()),
        scaffold_cap=args.scaffold_cap,
        pct_5ndz=round(100.0 * (selection.pocket == "5NDZ_allosteric").sum() / len(selection), 1),
        carve_lists=carve_counts,
        by_batch=selection.batch.value_counts().to_dict(),
        envelope=ENVELOPE,
        note="Diverse multi-axis selection. Ranking evidence only; no activity claim.",
    )
    (args.out_dir / "EX32_SELECTION_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
