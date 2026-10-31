#!/usr/bin/env python
"""
v3_freeze_denovo_seeds.py — STEP 1 of the sourceable-hit campaign.

Freeze a definitive, diversity-aware set of de novo similarity seeds (target
20-40) that collectively cover the winning PAR2 chemistry. These seeds are the
QUERIES for real-chemical-space retrieval; they are not themselves nominations.

Design rules (from the campaign brief and the 2026-09-05 handoff):
  * Do NOT simply take ranks 1..40. Selection is diversity-aware.
  * Prefer EX32-supported molecules, strong crystal-IFP recovery, strong
    hotspot score, defined stereochemistry, non-clashing poses, good ligand
    efficiency, sensible physchem, useful Fsp3, convergent chemotypes,
    novel scaffold hops, and reference/AZ8838-adjacent chemistry.
  * 5NDD orthosteric is primary; 5NDZ allosteric is a small hedge.
  * Scores are standardised WITHIN pocket. 5NDD and 5NDZ are never pooled.
  * Correlated scores are compressed into one evidence family, not counted
    as independent votes (Vina<->LE = -0.958, Vina<->PIGNet2 = 0.899,
    CNNscore<->CNN_VS = 0.988; IFP recovery correlates <=0.19 with all).
  * Missing values stay missing. Nothing is imputed to zero.

DESIGN_HOTSPOT_SCORE is defined here, explicitly, as the preparation-independent
polar-anchor evidence from v2_robust_anchors.py combined with pocket contact
breadth:
      0.60 * polar_anchor_fraction   (heavy-atom polar proximity to the
                                      crystallographic anchor residues, which
                                      is invariant to ligand protomer and
                                      receptor His tautomer)
    + 0.40 * contact_fraction        (fraction of the reference contact residue
                                      set that the pose touches)
It is a geometric descriptor. It is NOT a claim that a hydrogen bond exists and
NOT a probability of antagonism.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "YONSEI_FINAL10_V2" / "10_FINAL"

# Crystallographic reference antagonists (corrected tautomer for AZ8838).
REF = {
    "5NDD_orthosteric": ("AZ8838", "CC(Oc1cccc(F)c1)c1ncc[nH]1"),
    "5NDZ_allosteric": ("AZ3451", "COc1cc(Br)ccc1NC(=O)Nc1cccc(C(F)(F)F)c1"),
}


def morgan(smi: str):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def zscore(s: pd.Series) -> pd.Series:
    """Standardise, preserving NaN. Constant columns -> NaN (no information)."""
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def nanmean_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean across available columns; NaN only where every column is NaN."""
    sub = df[[c for c in cols if c in df.columns]]
    if sub.shape[1] == 0:
        return pd.Series(np.nan, index=df.index)
    return sub.mean(axis=1, skipna=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=32)
    ap.add_argument("--n-5ndz", type=int, default=6,
                    help="allosteric hedge size (subset of --n-seeds)")
    ap.add_argument("--out", default=str(REPO / "YONSEI_FINAL10_V2" /
                                         "11_SOURCEABLE" / "seeds"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    cons = pd.read_csv(FINAL / "CONSENSUS_FINAL" / "CONSENSUS_WITH_PROVENANCE.csv",
                       low_memory=False)
    ex32 = pd.read_csv(FINAL / "DENOVO_EX32_ALL.csv", low_memory=False)
    anch = pd.read_csv(FINAL / "ROBUST_ANCHORS_BEST.csv", low_memory=False)
    ster = pd.read_csv(FINAL / "STEREO_REDOCK_LIST.csv", low_memory=False)
    ideal = pd.read_csv(FINAL / "IDEAL_DE_NOVO_TOP5.csv", low_memory=False)
    panel = pd.read_csv(FINAL / "FINAL_TOP20_DIVERSE_PAR2_CANDIDATES.csv",
                        low_memory=False)
    prov = {"inputs": {}, "generated_utc": datetime.now(timezone.utc).isoformat()}
    prov["inputs"]["consensus_rows"] = int(len(cons))
    prov["inputs"]["ex32_rows"] = int(len(ex32))
    prov["inputs"]["robust_anchor_rows"] = int(len(anch))

    # de novo only: DN = batch 1, DL = batch 2, RX = real compounds
    df = cons[cons.cand_id.str.match(r"^D[NL]\d")].copy()
    prov["denovo_ligand_pocket_rows"] = int(len(df))
    prov["denovo_unique_molecules"] = int(df.cand_id.nunique())

    # ------------------------------------------------------- merge evidence
    # EX32 supersedes EX8 for the molecules that were refined.
    ex32_keep = ["cand_id", "pocket", "best_vina_affinity", "best_CNNscore",
                 "best_CNNaffinity", "best_CNN_VS", "max_ifp_recovery",
                 "ligand_efficiency", "ifp_hbonds", "ifp_keys",
                 "n_poses_acceptable", "all_poses_clashing", "heavy_atoms"]
    e = ex32[[c for c in ex32_keep if c in ex32.columns]].copy()
    e["evidence_level"] = "EX32"
    df = df.merge(e, on=["cand_id", "pocket"], how="left", suffixes=("", "_ex32"))
    df["evidence_level"] = df["evidence_level"].fillna("EX8")

    for c in ["best_vina_affinity", "best_CNNscore", "best_CNNaffinity",
              "best_CNN_VS", "max_ifp_recovery", "ligand_efficiency",
              "ifp_hbonds", "n_poses_acceptable", "all_poses_clashing",
              "heavy_atoms"]:
        c32 = c + "_ex32"
        if c32 in df.columns:
            df[c] = df[c32].where(df[c32].notna(), df[c])
            df.drop(columns=[c32], inplace=True)

    # preparation-independent anchors (DESIGN_HOTSPOT_SCORE inputs)
    a = anch[["cand_id", "pocket", "polar_anchor_count", "polar_anchor_fraction",
              "min_receptor_contact_A", "n_receptor_atoms_within_4_5A",
              "dist_TYR82", "dist_HIS135", "dist_ASP228",
              "dist_TYR210", "dist_ASN158", "dist_CYS161"]].copy()
    df = df.merge(a, on=["cand_id", "pocket"], how="left")

    # stereochemistry
    st = ster[["cand_id", "severity", "stereo_status", "smiles_input",
               "docked_smiles_prepared"]].rename(
        columns={"smiles_input": "stereo_smiles_input",
                 "docked_smiles_prepared": "stereo_smiles_docked"})
    df = df.merge(st, on="cand_id", how="left")
    df["stereo_flag"] = np.where(
        df.severity.eq("WRONG_STEREOISOMER_DOCKED"), "WRONG_ISOMER_DOCKED",
        np.where(df.severity.eq("AMBIGUOUS_STEREOCHEMISTRY"), "AMBIGUOUS", "OK"))

    # ------------------------------------------------------------ hard gates
    g = {}
    g["start"] = len(df)
    df = df[df.smiles_prepared.notna()]; g["has_prepared_smiles"] = len(df)
    # clash: True is a hard fail. NaN (batch-1 rows lack the metric) is allowed
    # through but recorded, and is penalised later - missing stays missing.
    df["clash_unknown"] = df.all_poses_clashing.isna()
    df = df[~df.all_poses_clashing.eq(True)]; g["not_all_poses_clashing"] = len(df)
    df = df[df.best_vina_affinity < 0]; g["negative_vina_affinity"] = len(df)
    # a molecule docked as the WRONG stereoisomer cannot seed a search: its
    # docking evidence describes a different molecule.
    df = df[df.stereo_flag != "WRONG_ISOMER_DOCKED"]; g["stereo_not_known_wrong"] = len(df)
    prov["gates"] = g

    # --------------------------------------------- DESIGN_HOTSPOT_SCORE
    df["DESIGN_HOTSPOT_SCORE"] = (
        0.60 * df["polar_anchor_fraction"] + 0.40 * df["contact_fraction"])
    df["CRYSTAL_IFP_RECOVERY"] = df["max_ifp_recovery"]

    # ------------------------------------------- per-pocket evidence families
    parts = []
    for pocket, sub in df.groupby("pocket"):
        sub = sub.copy()
        # F1 interaction geometry (highest weight, the only orthogonal evidence)
        sub["z_ifp"] = zscore(sub["CRYSTAL_IFP_RECOVERY"])
        sub["z_hotspot"] = zscore(sub["DESIGN_HOTSPOT_SCORE"])
        sub["F1_interaction"] = nanmean_cols(sub, ["z_ifp", "z_hotspot"])
        # F2 pose physics - correlated scores COMPRESSED into one family
        sub["z_vina"] = zscore(-sub["best_vina_affinity"])
        sub["z_le"] = zscore(sub["ligand_efficiency"])
        sub["z_cnns"] = zscore(sub["best_CNNscore"])
        sub["z_cnna"] = zscore(sub["best_CNNaffinity"])
        sub["_vina_block"] = nanmean_cols(sub, ["z_vina", "z_le"])
        sub["_cnn_block"] = nanmean_cols(sub, ["z_cnns", "z_cnna"])
        sub["F2_physics"] = nanmean_cols(sub, ["_vina_block", "_cnn_block"])
        # F3 chemistry / topical plausibility
        env = pd.DataFrame(index=sub.index)
        env["mw"] = np.clip(1 - (sub.mw - 350).abs() / 250, 0, 1)
        env["clogp"] = np.clip(1 - (sub.clogp - 2.5).abs() / 3.5, 0, 1)
        env["tpsa"] = np.clip(1 - (sub.tpsa - 75).abs() / 75, 0, 1)
        env["rotb"] = np.clip(1 - np.clip(sub.rotb - 8, 0, None) / 8, 0, 1)
        env["fsp3"] = np.clip(sub.fsp3 / 0.6, 0, 1)
        env["qed"] = sub.qed
        sub["F3_chemistry"] = env.mean(axis=1, skipna=True)
        sub["z_chem"] = zscore(sub["F3_chemistry"])
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)

    # evidence-level and confidence adjustments (explicit, small, documented)
    df["ex32_bonus"] = np.where(df.evidence_level.eq("EX32"), 0.35, 0.0)
    df["clash_unknown_penalty"] = np.where(df.clash_unknown, -0.25, 0.0)
    df["stereo_penalty"] = np.where(df.stereo_flag.eq("AMBIGUOUS"), -0.20, 0.0)

    df["SEED_SCORE"] = (
        0.45 * df["F1_interaction"].fillna(df["F1_interaction"].min())
        + 0.30 * df["F2_physics"].fillna(df["F2_physics"].min())
        + 0.25 * df["z_chem"].fillna(df["z_chem"].min())
        + df["ex32_bonus"] + df["clash_unknown_penalty"] + df["stereo_penalty"])

    # ---------------------------------------------------- scaffold + novelty
    def murcko(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        try:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
        except Exception:
            return None

    df["murcko_recomputed"] = [murcko(s) for s in df.smiles_prepared]
    df["murcko_use"] = df.murcko_recomputed.fillna(df.murcko).fillna("")

    ref_fp = {k: morgan(v[1]) for k, v in REF.items()}
    fps, tan = [], []
    for smi, pk in zip(df.smiles_prepared, df.pocket):
        fp = morgan(smi)
        fps.append(fp)
        r = ref_fp.get(pk)
        tan.append(DataStructs.TanimotoSimilarity(fp, r)
                   if (fp is not None and r is not None) else np.nan)
    df["fp"] = fps
    df["tanimoto_to_reference_recomputed"] = tan
    df["novelty_band"] = pd.cut(
        df["tanimoto_to_reference_recomputed"],
        [-0.01, 0.15, 0.30, 1.01],
        labels=["novel_chemotype", "moderate_scaffold_hop",
                "reference_adjacent"])

    df = df[df.fp.notna()].copy()

    # ------------------------------------------------- diversity-aware pick
    forced = set(ideal.cand_id) | set(panel[panel.cand_id.str.match(r"^D[NL]\d")].cand_id)
    n_5ndz = args.n_5ndz
    n_5ndd = args.n_seeds - n_5ndz

    def pick(sub: pd.DataFrame, n: int, forced_ids: set[str]) -> pd.DataFrame:
        """Greedy: force-include, then maximise evidence subject to
        scaffold uniqueness and fingerprint dissimilarity to what is chosen."""
        sub = sub.sort_values("SEED_SCORE", ascending=False).copy()
        chosen, seen_scaffold, seen_cid = [], set(), set()

        def try_add(row, why, max_sim=0.70):
            if row.cand_id in seen_cid:
                return False
            sc = row.murcko_use
            if sc and sc in seen_scaffold:
                return False
            for c in chosen:
                if DataStructs.TanimotoSimilarity(row.fp, c["_row"].fp) > max_sim:
                    return False
            chosen.append({"_row": row, "why_selected": why})
            seen_cid.add(row.cand_id)
            if sc:
                seen_scaffold.add(sc)
            return True

        # pass 1: forced (IDEAL_DE_NOVO_TOP5 + de novo members of the panel)
        for _, r in sub[sub.cand_id.isin(forced_ids)].iterrows():
            try_add(r, "carried forward: prior IDEAL_TOP5 / FINAL_TOP20 member",
                    max_sim=1.01)
        # pass 2: archetype + novelty-band coverage
        for band in ["reference_adjacent", "moderate_scaffold_hop", "novel_chemotype"]:
            for _, r in sub[sub.novelty_band.eq(band)].iterrows():
                if len(chosen) >= n:
                    break
                if sum(1 for c in chosen
                       if c["_row"].novelty_band == band) >= max(2, n // 4):
                    break
                try_add(r, f"novelty-band coverage: {band}")
        # pass 3: EX32-supported, best evidence first
        for _, r in sub[sub.evidence_level.eq("EX32")].iterrows():
            if len(chosen) >= n:
                break
            try_add(r, "EX32-refined, high combined evidence")
        # pass 4: fill on evidence
        for _, r in sub.iterrows():
            if len(chosen) >= n:
                break
            try_add(r, "diversity fill on combined evidence")
        rows = pd.DataFrame([c["_row"] for c in chosen])
        rows["why_selected"] = [c["why_selected"] for c in chosen]
        return rows

    d5ndd = pick(df[df.pocket.eq("5NDD_orthosteric")], n_5ndd,
                 forced & set(df[df.pocket.eq("5NDD_orthosteric")].cand_id))
    d5ndz = pick(df[df.pocket.eq("5NDZ_allosteric")], n_5ndz,
                 forced & set(df[df.pocket.eq("5NDZ_allosteric")].cand_id))
    seeds = pd.concat([d5ndd, d5ndz], ignore_index=True)
    seeds = seeds.sort_values(["pocket", "SEED_SCORE"],
                              ascending=[True, False]).reset_index(drop=True)
    seeds.insert(0, "seed_id", [f"SEED{i+1:03d}" for i in range(len(seeds))])

    # cluster membership, for reporting
    cl = []
    for pk, tag in [("5NDD_orthosteric", "5NDD"), ("5NDZ_allosteric", "5NDZ")]:
        p = FINAL / f"CLUSTERS_FINAL_{tag}" / "DENOVO_CLUSTER_MEMBERS.csv"
        if p.exists():
            m = pd.read_csv(p, low_memory=False)
            col = next((c for c in m.columns if c.startswith("cluster_id")), None)
            if col:
                cl.append(m[["cand_id", col]].rename(columns={col: "cluster_id_final"})
                          .assign(pocket=pk))
    if cl:
        cl = pd.concat(cl, ignore_index=True).drop_duplicates(["cand_id", "pocket"])
        seeds = seeds.merge(cl, on=["cand_id", "pocket"], how="left")

    # ------------------------------------------------------------- write out
    cols = ["seed_id", "cand_id", "pocket", "evidence_level", "smiles_prepared",
            "stereo_smiles_input", "inchikey", "stereo_flag", "stereo_status",
            "severity", "murcko_use", "cluster_id_final", "denovo_family",
            "interaction_archetype" if "interaction_archetype" in seeds.columns
            else "structural_evidence",
            "novelty_band", "tanimoto_to_reference_recomputed",
            "CRYSTAL_IFP_RECOVERY", "DESIGN_HOTSPOT_SCORE",
            "polar_anchor_count", "polar_anchor_fraction", "contact_fraction",
            "anchor_hbond_residues", "ifp_hbonds", "ifp_keys",
            "best_vina_affinity", "best_CNNscore", "best_CNNaffinity",
            "best_CNN_VS", "ligand_efficiency", "heavy_atoms",
            "mw", "clogp", "tpsa", "hbd", "hba", "rotb", "fsp3",
            "formal_charge", "qed",
            "F1_interaction", "F2_physics", "F3_chemistry", "SEED_SCORE",
            "n_poses_acceptable", "clash_unknown", "why_selected"]
    cols = [c for c in cols if c in seeds.columns]
    seeds_out = seeds[cols].copy()
    seeds_out = seeds_out.rename(columns={"murcko_use": "murcko_scaffold"})
    seeds_out.to_csv(out / "FINAL_DE_NOVO_SIMILARITY_SEEDS.csv", index=False)

    prov["seeds_selected"] = int(len(seeds_out))
    prov["by_pocket"] = {k: int(v) for k, v in seeds_out.pocket.value_counts().items()}
    prov["by_evidence_level"] = {k: int(v) for k, v in
                                 seeds_out.evidence_level.value_counts().items()}
    prov["by_novelty_band"] = {str(k): int(v) for k, v in
                               seeds_out.novelty_band.value_counts().items()}
    prov["by_stereo_flag"] = {k: int(v) for k, v in
                              seeds_out.stereo_flag.value_counts().items()}
    prov["unique_murcko_scaffolds"] = int(seeds_out.murcko_scaffold.nunique())
    prov["forced_carried_forward"] = sorted(forced & set(seeds_out.cand_id))
    prov["forced_not_available"] = sorted(forced - set(seeds_out.cand_id))
    prov["hotspot_definition"] = ("0.60*polar_anchor_fraction + "
                                  "0.40*contact_fraction; geometric only")
    (out / "SEED_SELECTION_PROVENANCE.json").write_text(json.dumps(prov, indent=2))

    print(json.dumps(prov, indent=2))
    print(f"\nWROTE {out/'FINAL_DE_NOVO_SIMILARITY_SEEDS.csv'}  rows={len(seeds_out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
