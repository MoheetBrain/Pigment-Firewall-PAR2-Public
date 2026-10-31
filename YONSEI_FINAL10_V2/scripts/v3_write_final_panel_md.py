#!/usr/bin/env python
"""
v3_write_final_panel_md.py — render FINAL_SOURCEABLE_PAR2_TOP20.md.

Writes a per-candidate narrative from the committed CSV. Every number printed
here is read from the panel table; nothing is recomputed or invented, and a
missing value is printed as "not measured" rather than as a zero or a guess.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

CAVEAT = """
> **What this list is.** Computationally prioritised, commercially sourceable
> PAR2 candidates. Nothing here is a demonstrated PAR2 antagonist. Docking,
> interaction fingerprints and co-folding are structural hypotheses, not
> assays, and none of them can separate an antagonist from a biased agonist —
> the decisive experiment is the G12/13→RhoA counter-screen.
"""


def fmt(v, nd=3, suffix=""):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "not measured"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}{suffix}"
    return str(v)


def get(r, *names, default=None):
    for n in names:
        if n in r.index and not pd.isna(r[n]):
            return r[n]
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="")
    args = ap.parse_args()

    d = pd.read_csv(args.panel)
    L = []
    L.append("# FINAL SOURCEABLE PAR2 TOP 20\n")
    L.append(CAVEAT)
    L.append(f"\n**{len(d)} candidates** · "
             f"{d.murcko_scaffold.nunique() if 'murcko_scaffold' in d else '?'} unique Murcko scaffolds\n")

    if "pocket" in d:
        L.append("\n| pocket | n |\n|---|---|")
        for k, v in d.pocket.value_counts().items():
            L.append(f"| {k} | {v} |")
    if "stock_status" in d:
        L.append("\n| sourceability | n |\n|---|---|")
        for k, v in d.stock_status.value_counts().items():
            L.append(f"| {k} | {v} |")

    L.append("\n---\n\n## Candidates\n")
    for _, r in d.iterrows():
        name = get(r, "compound_name", "cand_id", default="(unnamed)")
        rank = get(r, "final_rank", default="?")
        L.append(f"### {rank}. {name}  \n")
        L.append(f"`{get(r,'cand_id')}` · "
                 f"{get(r,'pocket',default='?')} · "
                 f"{get(r,'source_lane',default='?')}\n")
        L.append(f"- **SMILES** `{get(r,'isomeric_SMILES','smiles',default='?')}`")
        L.append(f"- **InChIKey** `{get(r,'inchikey',default='?')}`")
        L.append(f"- **Sourceability** {get(r,'stock_status',default='not resolved')}"
                 f" · vendor {get(r,'vendor','best_vendor',default='UNKNOWN')}"
                 f" · cat {get(r,'catalogue_id','best_catalogue_id',default='UNKNOWN')}"
                 f" · price {get(r,'price',default='UNKNOWN')}"
                 f" · lead time {get(r,'lead_time',default='UNKNOWN')}")
        L.append(f"- **De novo parent** {get(r,'de_novo_parent',default='none')}"
                 f" (Morgan Tanimoto {fmt(get(r,'parent_similarity','Morgan_Tanimoto','seed_tanimoto'))})")
        L.append(f"- **EX32** Vina {fmt(get(r,'EX32_Vina','best_vina_affinity'),2)} · "
                 f"CNNscore {fmt(get(r,'CNNscore','best_CNNscore'))} · "
                 f"CNNaffinity {fmt(get(r,'CNNaffinity','best_CNNaffinity'))} · "
                 f"LE {fmt(get(r,'ligand_efficiency'))}")
        L.append(f"- **Interaction geometry** crystal-IFP recovery "
                 f"{fmt(get(r,'CRYSTAL_IFP_RECOVERY'))} · hotspot "
                 f"{fmt(get(r,'DESIGN_HOTSPOT_SCORE'))} · key contacts "
                 f"{get(r,'key_contacts','anchor_hbond_residues',default='not measured')}")
        L.append(f"- **Co-folding** confidence {fmt(get(r,'cofold_confidence'))} · "
                 f"binder prob {fmt(get(r,'cofold_binder_probability'))} · "
                 f"pocket occupancy {fmt(get(r,'pocket_occupancy'))} · "
                 f"sample consistency {fmt(get(r,'cofold_consistency_rmsd'),2,' Å')} · "
                 f"GNINA agreement {fmt(get(r,'cofold_GNINA_pose_RMSD'),2,' Å')}")
        L.append(f"- **Chemistry** MW {fmt(get(r,'mw','MW'),1)} · cLogP {fmt(get(r,'clogp'),2)} · "
                 f"TPSA {fmt(get(r,'tpsa'),1)} · HBD {get(r,'hbd',default='?')} · "
                 f"HBA {get(r,'hba',default='?')} · rotB {get(r,'rotb',default='?')} · "
                 f"Fsp3 {fmt(get(r,'fsp3'),2)} · charge {get(r,'formal_charge',default='?')} · "
                 f"QED {fmt(get(r,'qed'),2)}")
        L.append(f"- **Active-state** {get(r,'active_state_warning',default='not assessed')}")
        L.append(f"- **Human exposure** {get(r,'human_exposure_note','human_exposure',default='not established')}")
        ev = get(r, "n_evidence_families")
        thin = get(r, "thin_evidence")
        if ev is not None:
            L.append(f"- **Evidence breadth** {int(ev)}/6 families"
                     + (" — **thin evidence, treat with caution**" if thin is True else ""))
        why = get(r, "why_selected")
        if why:
            L.append(f"\n{why}\n")
        unc = get(r, "largest_uncertainty")
        if unc:
            L.append(f"*Largest uncertainty:* {unc}\n")
        L.append("")

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out} ({len(d)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
