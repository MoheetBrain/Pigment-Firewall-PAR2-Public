#!/usr/bin/env python
"""
v3_build_cofold_inputs.py — build constrained Boltz-2 co-folding YAMLs.

One YAML per (molecule, pocket, seed). The pocket is constrained ONLY by the
identity of the crystallographic contact residues; the GNINA pose is never
supplied. Co-folding must therefore produce an INDEPENDENT structural
hypothesis rather than being handed the answer.

Target: wild-type human PAR2 (UniProt P55085) residues 59-370. The
crystallography construct's BRIL/T4L fusions and seven thermostabilising
mutations are excluded, and so is the N-terminal tethered-ligand region
(1-58), which would otherwise compete for the orthosteric site.

Co-folding confidence is NOT a binding assay and cannot distinguish an
antagonist from a biased agonist.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "YONSEI_FINAL10_V2" / "11_SOURCEABLE" / "cofold" / "PAR2_COFOLD_TARGET.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", required=True,
                    help="CSV with columns: cofold_id, smiles, pocket")
    ap.add_argument("--out", required=True, help="directory for the YAMLs")
    ap.add_argument("--seeds", default="20260906,20260907,20260908",
                    help="comma-separated diffusion seeds")
    ap.add_argument("--msa", default="", help="optional cached .a3m path used "
                                              "on the remote machine")
    ap.add_argument("--max-distance", type=float, default=6.0)
    args = ap.parse_args()

    meta = json.loads(TARGET.read_text())
    seq = meta["sequence"]
    pocket_idx = meta["pocket_residues_boltz_index"]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mols = pd.read_csv(args.molecules)
    for col in ("cofold_id", "smiles", "pocket"):
        if col not in mols.columns:
            print(f"ERROR: --molecules is missing column '{col}'", file=sys.stderr)
            return 2

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    written, skipped = [], []
    for _, r in mols.iterrows():
        smi = str(r.smiles).strip()
        m = Chem.MolFromSmiles(smi)
        if m is None:
            skipped.append({"cofold_id": r.cofold_id, "reason": "unparseable SMILES"})
            continue
        pk = str(r.pocket).strip()
        if pk not in pocket_idx:
            skipped.append({"cofold_id": r.cofold_id, "reason": f"unknown pocket {pk}"})
            continue
        # canonical isomeric SMILES: stereochemistry is carried through, never dropped
        smi_iso = Chem.MolToSmiles(m, isomericSmiles=True)
        contacts = ", ".join(f"[A, {i}]" for i in pocket_idx[pk])
        msa_line = f"\n      msa: {args.msa}" if args.msa else ""
        for sd in seeds:
            name = f"{r.cofold_id}__{pk}__s{sd}"
            yaml = (
                "version: 1\n"
                "sequences:\n"
                "  - protein:\n"
                "      id: A\n"
                f"      sequence: {seq}{msa_line}\n"
                "  - ligand:\n"
                "      id: B\n"
                f"      smiles: '{smi_iso}'\n"
                "constraints:\n"
                "  - pocket:\n"
                "      binder: B\n"
                f"      contacts: [{contacts}]\n"
                f"      max_distance: {args.max_distance}\n"
                "properties:\n"
                "  - affinity:\n"
                "      binder: B\n"
            )
            (out / f"{name}.yaml").write_text(yaml)
            written.append({"cofold_id": r.cofold_id, "pocket": pk, "seed": sd,
                            "yaml": f"{name}.yaml", "smiles_used": smi_iso})

    pd.DataFrame(written).to_csv(out.parent / "COFOLD_JOB_MANIFEST.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out.parent / "COFOLD_SKIPPED.csv", index=False)
    print(json.dumps({"yamls_written": len(written),
                      "molecules": int(mols.shape[0]),
                      "seeds_per_molecule": len(seeds),
                      "skipped": len(skipped)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
