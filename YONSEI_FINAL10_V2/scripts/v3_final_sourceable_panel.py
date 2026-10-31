#!/usr/bin/env python
"""
v3_final_sourceable_panel.py — build FINAL_SOURCEABLE_PAR2_TOP20.

Integrates the five evidence lanes into one diversity-constrained panel of
REAL, SOURCEABLE molecules.

RANKING PHILOSOPHY
------------------
Scores are NOT blindly averaged. Evidence is grouped into six families; within
a family, correlated measurements are compressed so that one piece of evidence
counts once. Families are then combined with explicit, auditable weights.

  HARD GATES (a molecule failing any of these is out, no trade-off):
    * chemically valid structure
    * stereochemistry not known-wrong
    * no catastrophic steric clash
    * sourceable or realistically make-on-demand
    * a valid EX32 pose
    * no active-state warning severe enough to reject

  F1 PAR2 INTERACTION GEOMETRY  (highest weight - the only orthogonal evidence)
  F2 POSE / BINDING PHYSICS     (Vina/CNN/LE compressed into ONE family)
  F3 CO-FOLDING                 (confidence, consistency, occupancy, convergence)
  F4 CHEMISTRY / PLAUSIBILITY
  F5 SIMILARITY / PROVENANCE
  F6 EXPERIMENTAL ACTIONABILITY

The output score is NOT a probability of antagonism. Nothing here demonstrates
that any molecule binds or inhibits PAR2.

Missing values stay missing. A molecule is never credited for a measurement
that was not made; absent evidence is handled by renormalising over the
families that ARE present, and the count of contributing families is reported
per row so a thinly-evidenced molecule is visible as such.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[2]

FAMILY_WEIGHTS = {
    "F1_interaction_geometry": 0.30,
    "F2_pose_physics": 0.15,
    "F3_cofolding": 0.20,
    "F4_chemistry": 0.10,
    "F5_similarity_provenance": 0.10,
    "F6_actionability": 0.15,
}

SOURCEABLE_OK = {"IN_STOCK", "IN_STOCK_VENDOR_LISTED", "MAKE_ON_DEMAND"}


def z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def unit(s: pd.Series) -> pd.Series:
    """Min-max to [0,1], preserving NaN."""
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def nanmean(df: pd.DataFrame, cols) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1, skipna=True)


def first_present(df: pd.DataFrame, names, default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def morgan(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def murcko(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def load(path: str | Path, what: str, prov: dict) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        prov.setdefault("missing_inputs", []).append({"what": what, "path": str(p)})
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    prov.setdefault("inputs", {})[what] = {"path": str(p), "rows": int(len(df))}
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex32", required=True, help="REAL_ANALOGUES_EX32.csv")
    ap.add_argument("--analogues", required=True, help="SOURCEABLE_ANALOGUES_DEDUP.csv")
    ap.add_argument("--sourceability", default="", help="SOURCEABILITY_SUMMARY.csv")
    ap.add_argument("--cofold", default="", help="PAR2_COFOLD_FINALISTS.csv")
    ap.add_argument("--activestate", default="",
                    help="SOURCEABLE_ACTIVE_STATE_COUNTERSCREEN.csv")
    ap.add_argument("--benchmark", default="", help="PAR2_ENRICHMENT_METRICS json/csv")
    ap.add_argument("--seeds", default=str(REPO / "YONSEI_FINAL10_V2" /
                                           "11_SOURCEABLE" / "seeds" /
                                           "FINAL_DE_NOVO_SIMILARITY_SEEDS.csv"))
    ap.add_argument("--n-final", type=int, default=20)
    ap.add_argument("--max-per-cluster", type=int, default=2)
    ap.add_argument("--cluster-threshold", type=float, default=0.65,
                    help="ECFP4 Tanimoto above which two molecules are 'close'")
    ap.add_argument("--max-5ndz", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "family_weights": FAMILY_WEIGHTS}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ex32 = load(args.ex32, "ex32", prov)
    ana = load(args.analogues, "analogues", prov)
    src = load(args.sourceability, "sourceability", prov) if args.sourceability else pd.DataFrame()
    cof = load(args.cofold, "cofold", prov) if args.cofold else pd.DataFrame()
    act = load(args.activestate, "activestate", prov) if args.activestate else pd.DataFrame()
    seeds = load(args.seeds, "seeds", prov)

    if ex32.empty:
        print("FATAL: no EX32 evidence; cannot build a panel.", file=sys.stderr)
        return 2

    df = ex32.copy()
    idcol = next((c for c in ["cand_id", "analogue_id", "cofold_id", "id"]
                  if c in df.columns), None)
    if idcol is None:
        print("FATAL: EX32 table has no recognisable id column.", file=sys.stderr)
        return 2
    df = df.rename(columns={idcol: "cand_id"})

    smi_col = next((c for c in ["smiles_prepared", "isomeric_smiles", "smiles"]
                    if c in df.columns), None)
    if smi_col is None:
        print("FATAL: EX32 table has no SMILES column.", file=sys.stderr)
        return 2
    df["isomeric_SMILES"] = df[smi_col]

    # ---- merge the lanes on cand_id / inchikey ------------------------------
    def merge_lane(base, lane, name):
        if lane.empty:
            return base
        key = next((c for c in ["cand_id", "analogue_id", "cofold_id", "id"]
                    if c in lane.columns), None)
        if key:
            lane = lane.rename(columns={key: "cand_id"})
            dup = [c for c in lane.columns if c in base.columns and c != "cand_id"]
            return base.merge(lane.drop(columns=dup).drop_duplicates("cand_id"),
                              on="cand_id", how="left")
        if "inchikey" in lane.columns and "inchikey" in base.columns:
            dup = [c for c in lane.columns if c in base.columns and c != "inchikey"]
            return base.merge(lane.drop(columns=dup).drop_duplicates("inchikey"),
                              on="inchikey", how="left")
        prov.setdefault("unmergeable_lanes", []).append(name)
        return base

    for lane, nm in [(ana, "analogues"), (src, "sourceability"),
                     (cof, "cofold"), (act, "activestate")]:
        df = merge_lane(df, lane, nm)

    # ---- normalise the key fields we depend on -----------------------------
    df["CRYSTAL_IFP_RECOVERY"] = first_present(df, ["CRYSTAL_IFP_RECOVERY", "max_ifp_recovery"])
    df["DESIGN_HOTSPOT_SCORE"] = first_present(df, ["DESIGN_HOTSPOT_SCORE"])
    if df["DESIGN_HOTSPOT_SCORE"].isna().all():
        paf = first_present(df, ["polar_anchor_fraction"])
        cf = first_present(df, ["contact_fraction"])
        df["DESIGN_HOTSPOT_SCORE"] = 0.60 * paf + 0.40 * cf
    df["EX32_Vina"] = first_present(df, ["best_vina_affinity", "EX32_Vina"])
    df["CNNscore"] = first_present(df, ["best_CNNscore", "CNNscore"])
    df["CNNaffinity"] = first_present(df, ["best_CNNaffinity", "CNNaffinity"])
    df["ligand_efficiency"] = first_present(df, ["ligand_efficiency"])
    df["stock_status"] = first_present(
        df, ["stock_status", "sourceability_class", "classification"],
        default="REGISTERED_NOT_VERIFIED")
    df["active_state_warning"] = first_present(
        df, ["active_state_class", "active_state_warning", "classification"],
        default="INCONCLUSIVE")
    df["pocket"] = first_present(df, ["pocket"], default="5NDD_orthosteric")

    df["murcko_scaffold"] = [murcko(s) for s in df.isomeric_SMILES]
    df["fp"] = [morgan(s) for s in df.isomeric_SMILES]

    # ---- HARD GATES ---------------------------------------------------------
    g = {"start": int(len(df))}
    df = df[df.fp.notna()].copy(); g["chemically_valid"] = int(len(df))

    stereo = first_present(df, ["stereo_flag", "stereo_status", "severity"], default="OK")
    df = df[~stereo.astype(str).str.contains("WRONG", case=False, na=False)]
    g["stereo_not_known_wrong"] = int(len(df))

    clash = first_present(df, ["all_poses_clashing"], default=False)
    df = df[~clash.fillna(False).astype(bool)]; g["no_catastrophic_clash"] = int(len(df))

    df = df[df.EX32_Vina.notna() & (df.EX32_Vina < 0)]
    g["valid_ex32_pose"] = int(len(df))

    df["is_sourceable"] = df.stock_status.astype(str).str.upper().isin(SOURCEABLE_OK)
    n_src = int(df.is_sourceable.sum())
    if n_src >= args.n_final:
        df = df[df.is_sourceable]
    else:
        # Do not silently drop the gate: keep REGISTERED_NOT_VERIFIED rows but
        # mark them, so the shortfall is visible instead of being papered over.
        prov["sourceability_gate_relaxed"] = {
            "verified_sourceable": n_src, "required": args.n_final,
            "note": "fewer verified-sourceable molecules than panel slots; "
                    "REGISTERED_NOT_VERIFIED rows retained and flagged"}
        df = df[df.stock_status.astype(str).str.upper() != "NOT_SOURCEABLE"]
    g["sourceable_or_make_on_demand"] = int(len(df))

    df = df[~df.active_state_warning.astype(str).str.upper()
            .str.contains("ACTIVE/GB88 WARNING|ACTIVE_GB88_WARNING", regex=True, na=False)]
    g["no_severe_active_state_warning"] = int(len(df))
    prov["gates"] = g

    if df.empty:
        print("FATAL: every candidate failed the hard gates.", file=sys.stderr)
        (out / "PANEL_PROVENANCE.json").write_text(json.dumps(prov, indent=2))
        return 3

    # ---- EVIDENCE FAMILIES (standardised WITHIN pocket) --------------------
    parts = []
    for pk, sub in df.groupby("pocket"):
        sub = sub.copy()
        sub["F1_interaction_geometry"] = nanmean(
            sub.assign(_a=z(sub.CRYSTAL_IFP_RECOVERY), _b=z(sub.DESIGN_HOTSPOT_SCORE)),
            ["_a", "_b"])
        # correlated pose scores compressed: Vina-block and CNN-block, then merged
        sub["_vina_block"] = nanmean(
            sub.assign(_a=z(-sub.EX32_Vina), _b=z(sub.ligand_efficiency)), ["_a", "_b"])
        sub["_cnn_block"] = nanmean(
            sub.assign(_a=z(sub.CNNscore), _b=z(sub.CNNaffinity)), ["_a", "_b"])
        sub["F2_pose_physics"] = nanmean(sub, ["_vina_block", "_cnn_block"])
        # ---- F3 CO-FOLDING -------------------------------------------------
        # Deliberately weighted AWAY from the Boltz affinity head. That head is
        # a demonstrably weak discriminator on this target: it assigns the
        # crystallographic binder AZ8838 a binder probability of only 0.583 and
        # ranks several de novo designs above it, which is not credible.
        # Structural evidence - does it land in the pocket, does it agree with
        # the docked pose, do independent diffusion samples agree - is what the
        # AZ8838 validation actually demonstrated the model can do
        # (ligand RMSD 1.22 A, centroid displacement 0.69 A).
        cofold_terms = {
            "pocket_occupancy":  (z(first_present(sub, ["pocket_occupancy"])),        0.30),
            "gnina_convergence": (z(-first_present(sub, ["cofold_GNINA_pose_RMSD",
                                                         "cofold_gnina_rmsd"])),      0.25),
            "sample_consistency": (z(-first_present(sub, ["cofold_consistency_rmsd"])), 0.20),
            "confidence":        (z(first_present(sub, ["cofold_confidence"])),       0.15),
            "binder_probability": (z(first_present(sub, ["cofold_binder_probability"])), 0.07),
            "affinity":          (z(-first_present(sub, ["cofold_affinity_pred_value"])), 0.03),
        }
        # Renormalise over the terms actually measured. A missing term
        # contributes NOTHING - it is never imputed to zero, which would
        # silently score an unmeasured molecule as merely average.
        num = pd.Series(0.0, index=sub.index)
        den = pd.Series(0.0, index=sub.index)
        for _name, (vals, wt) in cofold_terms.items():
            ok = vals.notna()
            num = num.add((vals.where(ok, 0.0) * wt), fill_value=0.0)
            den = den.add(pd.Series(np.where(ok, wt, 0.0), index=sub.index),
                          fill_value=0.0)
        sub["F3_cofolding"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
        sub["n_cofold_terms"] = sum(v.notna().astype(int)
                                    for v, _w in cofold_terms.values())
        # chemistry envelope
        env = pd.DataFrame(index=sub.index)
        mw = first_present(sub, ["mw", "MW"]); env["mw"] = np.clip(1 - (mw - 350).abs() / 250, 0, 1)
        lp = first_present(sub, ["clogp", "cLogP", "logp"]); env["clogp"] = np.clip(1 - (lp - 2.5).abs() / 3.5, 0, 1)
        tp = first_present(sub, ["tpsa", "TPSA"]); env["tpsa"] = np.clip(1 - (tp - 75).abs() / 75, 0, 1)
        rb = first_present(sub, ["rotb", "rotatable_bonds"]); env["rotb"] = np.clip(1 - np.clip(rb - 8, 0, None) / 8, 0, 1)
        f3 = first_present(sub, ["fsp3", "Fsp3"]); env["fsp3"] = np.clip(f3 / 0.6, 0, 1)
        env["qed"] = first_present(sub, ["qed", "QED"])
        sub["F4_chemistry"] = z(env.mean(axis=1, skipna=True))
        # similarity / provenance to the de novo parent
        sim = first_present(sub, ["parent_similarity", "Morgan_Tanimoto",
                                  "best_tanimoto", "tanimoto"])
        mcs = first_present(sub, ["mcs_fraction", "MCS_fraction"])
        shp = first_present(sub, ["shape_similarity", "usrcat_similarity"])
        sub["F5_similarity_provenance"] = nanmean(
            sub.assign(_a=z(sim), _b=z(mcs), _c=z(shp)), ["_a", "_b", "_c"])
        # actionability
        act_s = pd.Series(0.0, index=sub.index)
        st = sub.stock_status.astype(str).str.upper()
        act_s = (act_s.mask(st.isin(["IN_STOCK", "IN_STOCK_VENDOR_LISTED"]), 1.0)
                      .mask(st.eq("MAKE_ON_DEMAND"), 0.6)
                      .mask(st.eq("REGISTERED_NOT_VERIFIED"), 0.2))
        # a molecule listed by many independent suppliers is more actionable
        nv = first_present(sub, ["n_vendors"])
        act_s = act_s + 0.25 * np.clip(np.log10(nv.fillna(0) + 1) / 2.0, 0, 1)
        he = first_present(sub, ["human_exposure_note", "human_exposure"], default="")
        approved = he.astype(str).str.contains("approved|clinical", case=False, na=False)
        sub["F6_actionability"] = z(act_s + 0.35 * approved.astype(float))
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)

    # ---- combine families, renormalising over those actually present -------
    fams = list(FAMILY_WEIGHTS)
    # A family that was not measured contributes neither numerator nor
    # denominator. Nothing is imputed: a missing measurement never becomes 0,
    # and a molecule is never credited for evidence that does not exist.
    W = pd.DataFrame({f: np.where(df[f].notna(), FAMILY_WEIGHTS[f], 0.0)
                      for f in fams}, index=df.index)
    contrib = W.sum(axis=1)
    num = sum(df[f].where(df[f].notna(), 0.0) * W[f] for f in fams)
    df["PANEL_SCORE"] = np.where(contrib > 0, num / contrib.replace(0, np.nan), np.nan)
    df["n_evidence_families"] = (W > 0).sum(axis=1)
    df["evidence_weight_covered"] = contrib / sum(FAMILY_WEIGHTS.values())
    # A molecule scored on very little evidence is flagged, not silently ranked
    # alongside a fully-evidenced one.
    df["thin_evidence"] = df["evidence_weight_covered"] < 0.60

    # ---- DIVERSITY-CONSTRAINED SELECTION -----------------------------------
    df = df.sort_values("PANEL_SCORE", ascending=False).reset_index(drop=True)
    chosen, scaff_count, n_5ndz = [], {}, 0
    for _, r in df.iterrows():
        if len(chosen) >= args.n_final:
            break
        if r.pocket == "5NDZ_allosteric" and n_5ndz >= args.max_5ndz:
            continue
        sc = r.murcko_scaffold or ""
        if sc and scaff_count.get(sc, 0) >= 1:
            continue
        close = sum(1 for c in chosen
                    if DataStructs.TanimotoSimilarity(r.fp, c.fp) >= args.cluster_threshold)
        if close >= args.max_per_cluster:
            continue
        chosen.append(r)
        if sc:
            scaff_count[sc] = scaff_count.get(sc, 0) + 1
        if r.pocket == "5NDZ_allosteric":
            n_5ndz += 1

    panel = pd.DataFrame(chosen).reset_index(drop=True)
    panel.insert(0, "final_rank", range(1, len(panel) + 1))
    panel = panel.drop(columns=[c for c in ["fp"] if c in panel.columns])

    panel.to_csv(out / "FINAL_SOURCEABLE_PAR2_TOP20.csv", index=False)
    prov["panel_size"] = int(len(panel))
    prov["unique_murcko_scaffolds"] = int(panel.murcko_scaffold.nunique())
    prov["by_pocket"] = {k: int(v) for k, v in panel.pocket.value_counts().items()}
    prov["by_stock_status"] = {k: int(v) for k, v in panel.stock_status.value_counts().items()}
    prov["median_evidence_families"] = float(panel.n_evidence_families.median())
    (out / "PANEL_PROVENANCE.json").write_text(json.dumps(prov, indent=2, default=str))
    print(json.dumps(prov, indent=2, default=str)[:3000])
    print(f"\nWROTE {out/'FINAL_SOURCEABLE_PAR2_TOP20.csv'} rows={len(panel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
