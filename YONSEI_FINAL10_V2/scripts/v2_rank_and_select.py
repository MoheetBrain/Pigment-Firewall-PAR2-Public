#!/usr/bin/env python3
"""Integrate V2 screen evidence and select a diverse candidate short list.

Combines, per ligand and per pocket: GNINA affinity, CNN pose/virtual-screen
scores, ligand efficiency, optional ProLIF recovery of the crystallographic
antagonist fingerprint, physicochemical plausibility, structural-alert flags,
and provenance. Ranks within each pocket separately, then performs a greedy
diversity-constrained selection.

The output is a prioritised short list for experimental triage. It is not a
statement that any molecule inhibits PAR2.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

POCKETS = ["5NDD_orthosteric", "5NDZ_allosteric"]

WEIGHTS = dict(cnn_vs=0.30, affinity=0.25, ifp=0.20, ligand_efficiency=0.15, drug_likeness=0.10)

REACTIVE_SMARTS = {
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "michael_acceptor_aldehyde": "[CX3H1](=O)[CX3]=[CX3]",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "azide": "[N-]=[N+]=[N-]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "nitroso": "[NX2]=[OX1]",
    "peroxide": "[OX2][OX2]",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "aliphatic_halide_reactive": "[CX4;!$(C(F)(F)F)][Cl,Br,I]",
    "thiol": "[SX2H1]",
    "quinone": "O=C1C=CC(=O)C=C1",
}


def robust_z(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    scale = mad * 1.4826 if mad and mad > 1e-9 else (values.std() or 1.0)
    z = (values - median) / scale
    if not higher_is_better:
        z = -z
    return z.clip(-4, 4)


def build_alert_flags(smiles_list: list[str]) -> pd.DataFrame:
    def catalog_for(entry) -> FilterCatalog:
        params = FilterCatalogParams()
        params.AddCatalog(entry)
        return FilterCatalog(params)

    pains_catalog = catalog_for(FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_catalog = catalog_for(FilterCatalogParams.FilterCatalogs.BRENK)
    patterns = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE_SMARTS.items()}
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            rows.append(dict(pains=True, brenk=True, structural_alerts="UNPARSABLE",
                             reactive_groups="UNPARSABLE"))
            continue
        pains_hits = [e.GetDescription() for e in pains_catalog.GetMatches(mol)]
        brenk_hits = [e.GetDescription() for e in brenk_catalog.GetMatches(mol)]
        reactive = [name for name, patt in patterns.items()
                    if patt is not None and mol.HasSubstructMatch(patt)]
        rows.append(dict(
            pains=bool(pains_hits),
            brenk=bool(brenk_hits),
            structural_alerts=";".join(sorted(set(pains_hits + brenk_hits))[:8]),
            reactive_groups=";".join(reactive),
        ))
    return pd.DataFrame(rows)


def fingerprints(smiles_list: list[str]):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        out.append(generator.GetFingerprint(mol) if mol is not None else None)
    return out


def diverse_select(frame: pd.DataFrame, n_final: int, tanimoto_cutoff: float,
                   max_per_scaffold: int) -> pd.DataFrame:
    frame = frame.sort_values("composite_score", ascending=False).reset_index(drop=True)
    fps = fingerprints(frame.smiles_prepared.tolist())
    selected: list[int] = []
    scaffold_counts: dict[str, int] = {}
    rejected: dict[int, str] = {}
    for i in range(len(frame)):
        if len(selected) >= n_final:
            break
        scaffold = frame.murcko.iloc[i] if isinstance(frame.murcko.iloc[i], str) else ""
        if scaffold and scaffold_counts.get(scaffold, 0) >= max_per_scaffold:
            rejected[i] = "scaffold_quota"
            continue
        fp = fps[i]
        if fp is None:
            rejected[i] = "unparsable"
            continue
        too_similar = False
        for j in selected:
            if fps[j] is None:
                continue
            if DataStructs.TanimotoSimilarity(fp, fps[j]) > tanimoto_cutoff:
                too_similar = True
                break
        if too_similar:
            rejected[i] = "similarity_cutoff"
            continue
        selected.append(i)
        if scaffold:
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
    out = frame.iloc[selected].copy()
    out.insert(0, "final_rank", range(1, len(out) + 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ligand-table", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="V2_LIGAND_POCKET_SCORES.csv")
    parser.add_argument("--interactions", type=Path, default=None,
                        help="optional V2_POSE_INTERACTIONS_SUMMARY.csv from v2_interactions.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-final", type=int, default=30)
    parser.add_argument("--tanimoto-cutoff", type=float, default=0.60)
    parser.add_argument("--max-per-scaffold", type=int, default=2)
    parser.add_argument("--max-clogp", type=float, default=5.0)
    parser.add_argument("--max-mw", type=float, default=520.0)
    parser.add_argument("--drop-alerts", action="store_true",
                        help="exclude PAINS/reactive hits from the final selection instead of flagging")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ligands = pd.read_csv(args.ligand_table)
    scores = pd.read_csv(args.scores)
    merged = scores.merge(ligands, on="cand_id", how="left", suffixes=("", "_lig"))
    if "heavy_atoms_lig" in merged:
        merged["heavy_atoms"] = merged["heavy_atoms"].fillna(merged["heavy_atoms_lig"])

    if args.interactions and args.interactions.exists():
        ifp = pd.read_csv(args.interactions)
        merged = merged.merge(ifp[["cand_id", "pocket", "ifp_recovery", "ifp_n_interactions"]],
                              on=["cand_id", "pocket"], how="left")
    else:
        merged["ifp_recovery"] = np.nan
        merged["ifp_n_interactions"] = np.nan

    alerts = build_alert_flags(merged.smiles_prepared.tolist())
    merged = pd.concat([merged.reset_index(drop=True), alerts.reset_index(drop=True)], axis=1)

    parts = []
    for pocket, sub in merged.groupby("pocket"):
        sub = sub.copy()
        sub["z_cnn_vs"] = robust_z(sub.best_CNN_VS, True)
        sub["z_affinity"] = robust_z(sub.best_affinity, False)
        sub["z_ligand_efficiency"] = robust_z(sub.ligand_efficiency, True)
        sub["z_qed"] = robust_z(sub.qed, True)
        if sub.ifp_recovery.notna().any():
            sub["z_ifp"] = robust_z(sub.ifp_recovery, True)
            ifp_weight = WEIGHTS["ifp"]
        else:
            sub["z_ifp"] = 0.0
            ifp_weight = 0.0
        renorm = 1.0 / (sum(WEIGHTS.values()) - WEIGHTS["ifp"] + ifp_weight)
        sub["evidence_score"] = renorm * (
            WEIGHTS["cnn_vs"] * sub.z_cnn_vs
            + WEIGHTS["affinity"] * sub.z_affinity
            + WEIGHTS["ligand_efficiency"] * sub.z_ligand_efficiency
            + WEIGHTS["drug_likeness"] * sub.z_qed
            + ifp_weight * sub.z_ifp
        )
        penalty = np.zeros(len(sub))
        penalty += 0.35 * (sub.clogp > args.max_clogp).to_numpy()
        penalty += 0.35 * (sub.mw > args.max_mw).to_numpy()
        penalty += 0.50 * sub.pains.fillna(False).to_numpy()
        penalty += 0.30 * (sub.reactive_groups.fillna("") != "").to_numpy()
        penalty += 0.20 * (sub.formal_charge.abs() >= 2).to_numpy()
        sub["penalty"] = penalty
        sub["composite_score"] = sub.evidence_score - penalty
        sub["pocket_rank"] = sub.composite_score.rank(ascending=False, method="first").astype(int)
        parts.append(sub)
    ranked = pd.concat(parts, ignore_index=True)
    ranked = ranked.sort_values("composite_score", ascending=False)
    ranked.to_csv(out / "V2_RANKED_ALL.csv", index=False)

    for pocket in POCKETS:
        sub = ranked[ranked.pocket == pocket]
        if not sub.empty:
            sub.head(100).to_csv(out / f"V2_TOP100_{pocket}.csv", index=False)

    # One row per ligand: keep its best-scoring pocket, record the other.
    best_per_ligand = ranked.sort_values("composite_score", ascending=False).drop_duplicates("cand_id")
    other = ranked.sort_values("composite_score", ascending=False).groupby("cand_id").nth(1)
    best_per_ligand = best_per_ligand.merge(
        other[["cand_id", "pocket", "composite_score", "best_affinity"]].rename(
            columns={"pocket": "second_pocket", "composite_score": "second_pocket_composite",
                     "best_affinity": "second_pocket_affinity"}),
        on="cand_id", how="left")
    best_per_ligand["pocket_selectivity"] = (
        best_per_ligand.composite_score - best_per_ligand.second_pocket_composite)

    pool = best_per_ligand
    if args.drop_alerts:
        pool = pool[(~pool.pains.fillna(False)) & (pool.reactive_groups.fillna("") == "")]
    pool.head(500).to_csv(out / "V2_TOP500_LIGANDS.csv", index=False)
    pool.head(100).to_csv(out / "V2_TOP100_LIGANDS.csv", index=False)

    final = diverse_select(pool, args.n_final, args.tanimoto_cutoff, args.max_per_scaffold)
    final.to_csv(out / f"V2_FINAL_{args.n_final}.csv", index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        ligand_pocket_rows=int(len(ranked)),
        unique_ligands=int(ranked.cand_id.nunique()),
        interactions_used=bool(args.interactions and args.interactions.exists()),
        weights=WEIGHTS,
        selection=dict(n_final=args.n_final, tanimoto_cutoff=args.tanimoto_cutoff,
                       max_per_scaffold=args.max_per_scaffold, drop_alerts=args.drop_alerts),
        final_by_pocket={k: int(v) for k, v in final.pocket.value_counts().items()},
        final_by_source={k: int(v) for k, v in final.source.value_counts().items()},
        final_unique_scaffolds=int(final.murcko.nunique()),
        final_composite_range=[float(final.composite_score.min()), float(final.composite_score.max())],
        caveat=("Docking, CNN rescoring and interaction recovery are ranking evidence only. "
                "No molecule here is demonstrated to bind or inhibit PAR2. Vendor and price "
                "fields remain PENDING until a live catalogue lookup is performed."),
    )
    (out / "V2_SELECTION_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
