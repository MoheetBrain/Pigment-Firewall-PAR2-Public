#!/usr/bin/env python3
"""Build the diverse final PAR2 candidate panel and the ideal de novo short list.

Selection is hierarchical, not a scalar sum:

  hard validity gates -> evidence families -> diversity-aware pick

Gates a molecule must pass: parseable structure, a non-clashing pose, correct
pocket routing, real GNINA numbers, no reactive/PAINS liability, and not a
molecule whose docked stereoisomer is known to be wrong.

Evidence families are kept separate because they are NOT independent: within
5NDD, Vina and ligand efficiency correlate -0.958, Vina and PIGNet2 0.899, and
CNNscore and CNN_VS 0.988, while interaction geometry correlates below 0.19 with
all of them. Diversity then spreads the panel across clusters, scaffolds,
pockets and interaction archetypes so a weaker but distinct hypothesis can beat
the seventh analogue of the best scaffold.

Nothing here asserts PAR2 antagonism. Static docking cannot distinguish an
antagonist from a biased agonist.
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

RDLogger.DisableLog("rdApp.*")
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ARCHETYPES = [
    ("reference_adjacent", "Tanimoto to the pocket's crystal antagonist >= 0.35"),
    ("triple_anchor", "polar contact to all three of TYR:82, HIS:135, ASP:228"),
    ("dual_anchor", "polar contact to two of the three anchors"),
    ("crystal_contact_reproducer", "crystal fingerprint recovery >= 0.85"),
    ("high_sp3_wedge", "Fsp3 >= 0.55 with a non-planar core"),
    ("compact_efficient", "<= 22 heavy atoms with ligand efficiency >= 0.38"),
    ("topical_envelope", "inside the MW/cLogP/TPSA envelope for topical use"),
    ("allosteric_hedge", "5NDZ extrahelical site"),
]


def archetype(row) -> str:
    tags = []
    if row.get("tanimoto_to_reference", 0) and row["tanimoto_to_reference"] >= 0.35:
        tags.append("reference_adjacent")
    frac = row.get("anchor_hbond_fraction")
    if pd.notna(frac):
        if frac >= 0.99:
            tags.append("triple_anchor")
        elif frac >= 0.66:
            tags.append("dual_anchor")
    if row.get("max_ifp_recovery", 0) and row["max_ifp_recovery"] >= 0.85:
        tags.append("crystal_contact_reproducer")
    if row.get("fsp3", 0) and row["fsp3"] >= 0.55:
        tags.append("high_sp3_wedge")
    if (row.get("heavy_atoms", 99) or 99) <= 22 and (row.get("ligand_efficiency", 0) or 0) >= 0.38:
        tags.append("compact_efficient")
    if (350 <= (row.get("mw") or 0) <= 480 and 1.5 <= (row.get("clogp") or -9) <= 3.0
            and 70 <= (row.get("tpsa") or 0) <= 95):
        tags.append("topical_envelope")
    if row.get("pocket") == "5NDZ_allosteric":
        tags.append("allosteric_hedge")
    return ";".join(tags) if tags else "general_binder"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, nargs="+", required=True)
    ap.add_argument("--stereo", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-final", type=int, default=20)
    ap.add_argument("--max-per-cluster", type=int, default=2)
    ap.add_argument("--tanimoto-cutoff", type=float, default=0.65)
    args = ap.parse_args()

    d = pd.read_csv(args.consensus)
    clusters = pd.concat([pd.read_csv(p) for p in args.clusters], ignore_index=True)
    keep = [c for c in ["cand_id", "pocket", "cluster_id"] if c in clusters]
    d = d.merge(clusters[keep].drop_duplicates(["cand_id", "pocket"]),
                on=["cand_id", "pocket"], how="left", suffixes=("", "_cl"))
    if "cluster_id_cl" in d:
        d["cluster_id"] = d.cluster_id_cl.fillna(d.get("cluster_id"))

    wrong_stereo = set()
    if args.stereo and args.stereo.exists():
        st = pd.read_csv(args.stereo)
        wrong_stereo = set(st[st.severity == "WRONG_STEREOISOMER_DOCKED"].cand_id)
        ambiguous = set(st[st.severity == "AMBIGUOUS_STEREOCHEMISTRY"].cand_id)
    else:
        ambiguous = set()
    d["stereo_flag"] = np.where(d.cand_id.isin(wrong_stereo), "WRONG_STEREOISOMER_DOCKED",
                        np.where(d.cand_id.isin(ambiguous), "AMBIGUOUS", "OK"))

    n0 = len(d)
    gates = {}
    g = d.copy()
    g = g[g.smiles_prepared.notna()];                        gates["parseable"] = len(g)
    # Coerce explicitly: after merging lanes this column can arrive as object
    # dtype, where `~` performs integer inversion instead of logical negation.
    clash = (g["all_poses_clashing"].fillna(False).astype(bool)
             if "all_poses_clashing" in g.columns else pd.Series(False, index=g.index))
    g = g[~clash]; gates["has_non_clashing_pose"] = len(g)
    g = g[g.best_vina_affinity.notna() & (g.best_vina_affinity < 0)]; gates["real_negative_affinity"] = len(g)
    if "pains_flag" in g:
        g = g[~g.pains_flag.fillna(False).astype(bool)];      gates["no_PAINS"] = len(g)
    if "reactive_groups" in g:
        g = g[g.reactive_groups.fillna("") == ""];            gates["no_reactive_group"] = len(g)
    g = g[g.stereo_flag != "WRONG_STEREOISOMER_DOCKED"];      gates["stereo_not_known_wrong"] = len(g)
    print(f"[gates] {n0} rows -> " + " -> ".join(f"{k} {v}" for k, v in gates.items()))

    g["interaction_archetype"] = [archetype(r._asdict()) for r in g.itertuples()]
    g["primary_archetype"] = g.interaction_archetype.str.split(";").str[0]
    # Real compounds carry their own provenance lane; de novo designs do not.
    lane = np.where(g.batch == "ex32", "DE_NOVO_EX32_REFINED", "DE_NOVO_EX8")
    if "source_lane_prov" in g.columns:
        lane = np.where(g.source_lane_prov.notna(), g.source_lane_prov, lane)
    g["source_lane"] = lane
    g["evidence_level"] = np.where(g.batch == "ex32", "EX32", "EX8")

    g = g.sort_values(["evidence_level", "consensus_score"], ascending=[True, False])
    fps, order = [], list(g.index)
    for s in g.smiles_prepared:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        fps.append(MORGAN.GetFingerprint(m) if m else None)
    fpmap = dict(zip(order, fps))

    selected, per_cluster, per_scaffold, per_arch, per_pocket = [], {}, {}, {}, {}
    floor_5ndz = max(2, args.n_final // 6)

    def admissible(idx, row, enforce_arch=True):
        cid = row.get("cluster_id")
        key = f"{row.pocket}:{cid}"
        if per_cluster.get(key, 0) >= args.max_per_cluster:
            return False
        scaf = row.get("murcko") or ""
        if scaf and per_scaffold.get(scaf, 0) >= 1:
            return False
        if enforce_arch and per_arch.get(row.primary_archetype, 0) >= max(3, args.n_final // 4):
            return False
        fp = fpmap.get(idx)
        if fp is not None and any(
                fpmap.get(j) is not None
                and DataStructs.TanimotoSimilarity(fp, fpmap[j]) > args.tanimoto_cutoff
                for j in selected):
            return False
        return True

    def take(idx, row):
        selected.append(idx)
        key = f"{row.pocket}:{row.get('cluster_id')}"
        per_cluster[key] = per_cluster.get(key, 0) + 1
        per_arch[row.primary_archetype] = per_arch.get(row.primary_archetype, 0) + 1
        per_pocket[row.pocket] = per_pocket.get(row.pocket, 0) + 1
        scaf = row.get("murcko") or ""
        if scaf:
            per_scaffold[scaf] = per_scaffold.get(scaf, 0) + 1

    # Pass 0: RESERVE slots for real, already-existing chemistry. De novo rows
    # carry EX32 evidence and therefore sort ahead of every real compound, which
    # only has EX8. Without a reservation the panel fills entirely with molecules
    # that must be synthesised, and the repurposing hypothesis - the lane most
    # likely to yield something testable soon - never appears at all.
    real_quota = max(4, args.n_final // 4)
    real_rows = g[~g.source_lane.str.startswith("DE_NOVO")]
    for idx, row in real_rows.iterrows():
        if sum(1 for j in selected if not g.loc[j, "source_lane"].startswith("DE_NOVO")) >= real_quota:
            break
        if admissible(idx, row):
            take(idx, row)

    # Pass 1: RESERVE slots for the 5NDZ allosteric hedge so the orthosteric lane,
    # which dominates on score, cannot crowd the second mechanism out entirely.
    for idx, row in g[g.pocket == "5NDZ_allosteric"].iterrows():
        if per_pocket.get("5NDZ_allosteric", 0) >= floor_5ndz:
            break
        if admissible(idx, row):
            take(idx, row)

    # Pass 2: fill the remainder by evidence level then consensus score.
    for enforce_arch in (True, False):
        for idx, row in g.iterrows():
            if len(selected) >= args.n_final:
                break
            if idx in selected:
                continue
            if admissible(idx, row, enforce_arch):
                take(idx, row)
        if len(selected) >= args.n_final:
            break

    panel = g.loc[selected].copy().reset_index(drop=True)
    panel.insert(0, "FINAL_RANK", range(1, len(panel) + 1))
    for col, val in (("vendor", "PENDING"), ("catalogue_ID", "PENDING"),
                     ("approximate_price", "PENDING"), ("stock_status", "PENDING")):
        panel[col] = val
    panel["human_exposure_note"] = np.where(
        panel.source_lane == "APPROVED_REPURPOSING",
        "approved drug - established human exposure; sourceable in principle",
        np.where(panel.source_lane == "REAL_ANALOGUE_PUBCHEM",
                 "registered compound - human exposure not established",
                 np.where(panel.source_lane == "NATURAL_PRODUCT_OR_V1_HIT",
                          "natural product - human exposure not established",
                          "de novo design - no human exposure; requires synthesis")))
    panel["mechanism_hypothesis"] = np.where(
        panel.pocket == "5NDD_orthosteric",
        "competitive orthosteric wedge; blocks tethered-ligand docking",
        "extrahelical negative allosteric modulation of TM2-4")
    panel["major_uncertainty"] = np.where(
        panel.stereo_flag == "AMBIGUOUS",
        "stereochemistry unassigned in the design; docked isomer is one of several",
        "antagonist vs biased agonist undetermined by docking")
    panel["what_this_teaches"] = panel.primary_archetype.map(lambda a: {
        "triple_anchor": "whether satisfying all three polar anchors is sufficient for blockade",
        "dual_anchor": "whether two of three anchors suffice",
        "crystal_contact_reproducer": "whether reproducing the AZ8838 contact set predicts activity",
        "reference_adjacent": "whether AZ8838-like chemistry reproduces in our hands",
        "high_sp3_wedge": "whether 3D sp3-rich cores improve on the flat reference chemotype",
        "compact_efficient": "whether a small efficient binder can hold the occluded subpocket",
        "topical_envelope": "whether the topical property envelope is compatible with binding",
        "allosteric_hedge": "whether the allosteric site is a viable fallback mechanism",
    }.get(a, "general structure-activity information"))

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["FINAL_RANK", "cand_id", "smiles_prepared", "inchikey", "stereo_flag",
            "source_lane", "evidence_level", "vendor", "catalogue_ID", "stock_status",
            "approximate_price", "pocket", "mechanism_hypothesis", "cluster_id", "murcko",
            "interaction_archetype", "primary_archetype", "best_vina_affinity",
            "best_CNNscore", "best_CNNaffinity", "best_CNN_VS", "pignet2_best",
            "max_ifp_recovery", "anchor_hbond_fraction", "anchor_hbond_residues",
            "contact_fraction", "ifp_hbonds", "ligand_efficiency", "heavy_atoms",
            "mw", "clogp", "tpsa", "hbd", "hba", "rotb", "fsp3", "formal_charge", "qed",
            "profile_fit", "tanimoto_to_reference", "reference_antagonist",
            "human_exposure_note", "consensus_score", "major_uncertainty",
            "what_this_teaches", "denovo_family", "batch"]
    cols = [c for c in cols if c in panel.columns]
    panel[cols].to_csv(args.out / f"FINAL_TOP{args.n_final}_DIVERSE_PAR2_CANDIDATES.csv", index=False)

    ideal = g[(g.pocket == "5NDD_orthosteric")].copy()
    ideal["ideal_score"] = (ideal.max_ifp_recovery.fillna(0) * 0.35
                            + ideal.anchor_hbond_fraction.fillna(0) * 0.30
                            + ideal.profile_fit.fillna(0) * 0.20
                            + ideal.fsp3.fillna(0) * 0.15)
    ideal = ideal.sort_values(["evidence_level", "ideal_score"], ascending=[True, False])
    ideal = ideal.drop_duplicates("murcko").head(5).reset_index(drop=True)
    ideal.insert(0, "IDEAL_RANK", range(1, len(ideal) + 1))
    ideal[[c for c in cols + ["ideal_score"] if c in ideal.columns]].to_csv(
        args.out / "IDEAL_DE_NOVO_TOP5.csv", index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        rows_considered=n0, gates=gates, panel_size=int(len(panel)),
        unique_murcko_scaffolds=int(panel.murcko.nunique()),
        unique_clusters=int(panel.cluster_id.nunique()),
        interaction_archetypes=int(panel.primary_archetype.nunique()),
        archetype_counts=panel.primary_archetype.value_counts().to_dict(),
        by_pocket=panel.pocket.value_counts().to_dict(),
        by_evidence_level=panel.evidence_level.value_counts().to_dict(),
        by_source_lane=panel.source_lane.value_counts().to_dict(),
        stereo_flags=panel.stereo_flag.value_counts().to_dict(),
        requiring_synthesis=int((panel.source_lane.str.startswith("DE_NOVO")).sum()),
        real_registered_compounds=int((~panel.source_lane.str.startswith("DE_NOVO")).sum()),
        approved_or_clinical=int((panel.source_lane == "APPROVED_REPURPOSING").sum()),
        ideal_top5=ideal.cand_id.tolist(),
        caveat=("Computationally prioritised PAR2 candidates, not demonstrated "
                "antagonists. Static docking cannot separate an antagonist from a "
                "biased agonist; the decisive counter-screen is G12/13 -> RhoA."),
    )
    (args.out / "FINAL_PANEL_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
