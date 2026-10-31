#!/usr/bin/env python
"""
v3_prepare_finalists.py — ligand preparation for the sourceable finalists.

Identical to the campaign's `v2_prepare_ligands.py` EXCEPT that it does NOT call
`rdMolStandardize.TautomerEnumerator().Canonicalize()`.

That call is still live at v2_prepare_ligands.py:132 and is the defect recorded
in HANDOFF_2026-09-05.md section 4.7: it destroyed stereochemistry in 2,534
designs, changed the constitution of 1,199 (mostly enol->keto), and invented new
unassigned stereocentres in 2,241. 207 ranked finalists were consequently docked
as the wrong stereoisomer.

These molecules are ones we intend to ORDER. Buying the wrong stereoisomer
wastes an assay slot and the purchase order looks correct while doing it, so
stereochemistry is preserved here and verified per molecule after preparation.

The pH 7.4 protonation is imported unchanged from the campaign script so that
ionisation is identical.

CAVEAT, recorded deliberately: skipping tautomer canonicalisation means these
ligands are not prepared byte-identically to the campaign's EX8 rows. EX32
values produced here are internally consistent across all 80 finalists and are
ranked within that set; they are NOT interchangeable with campaign EX8 numbers
for the same molecule.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_prepare_ligands import protonate_ph74   # identical ionisation rules

RDLogger.DisableLog("rdApp.*")


def n_defined_stereo(mol: Chem.Mol) -> int:
    if mol is None:
        return -1
    return sum(1 for a in mol.GetAtoms() if a.HasProp("_CIPCode"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--shard-size", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out); (out / "shards").mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(args.queue)

    lf = rdMolStandardize.LargestFragmentChooser()
    rows, failures, stereo_changed = [], [], []
    per_pocket: dict[str, list] = {}

    for _, r in q.iterrows():
        cid, smi, pocket = str(r.cand_id), str(r.smiles), str(r.pocket)
        m0 = Chem.MolFromSmiles(smi)
        if m0 is None:
            failures.append({"cand_id": cid, "reason": "unparseable input SMILES"})
            continue
        Chem.AssignStereochemistry(m0, cleanIt=True, force=True)
        n_in = n_defined_stereo(m0)
        try:
            m = rdMolStandardize.Cleanup(m0)
            m = lf.choose(m)
            # NO TautomerEnumerator().Canonicalize() - see module docstring.
            m = protonate_ph74(m)
            m = Chem.AddHs(m)
            params = AllChem.ETKDGv3(); params.randomSeed = args.seed
            if AllChem.EmbedMolecule(m, params) != 0:
                failures.append({"cand_id": cid, "reason": "embed failed"}); continue
            AllChem.MMFFOptimizeMolecule(m, maxIters=400)
        except Exception as e:
            failures.append({"cand_id": cid, "reason": f"prep error: {e}"}); continue

        mh = Chem.RemoveHs(Chem.Mol(m))
        Chem.AssignStereochemistry(mh, cleanIt=True, force=True)
        n_out = n_defined_stereo(mh)
        if n_out < n_in:
            stereo_changed.append({"cand_id": cid, "stereocentres_in": n_in,
                                   "stereocentres_out": n_out,
                                   "smiles_in": smi,
                                   "smiles_out": Chem.MolToSmiles(mh)})
        m.SetProp("_Name", cid)
        per_pocket.setdefault(pocket, []).append(m)
        rows.append({"cand_id": cid, "pocket": pocket,
                     "smiles_input": smi,
                     "smiles_prepared": Chem.MolToSmiles(mh),
                     "inchikey_prepared": Chem.MolToInchiKey(mh),
                     "stereocentres_in": n_in, "stereocentres_out": n_out,
                     "stereo_preserved": bool(n_out >= n_in),
                     "formal_charge": Chem.GetFormalCharge(mh),
                     "heavy_atoms": mh.GetNumHeavyAtoms()})

    manifest = {"queue": str(args.queue), "seed": args.seed,
                "prepared": len(rows), "failed": len(failures),
                "stereo_lost": len(stereo_changed),
                "tautomer_canonicalisation": "DISABLED (see docstring)",
                "shards": {}}
    for pocket, mols in per_pocket.items():
        for i in range(0, len(mols), args.shard_size):
            chunk = mols[i:i + args.shard_size]
            name = f"{pocket}_shard{i//args.shard_size:04d}.sdf"
            w = Chem.SDWriter(str(out / "shards" / name))
            for mm in chunk:
                w.write(mm)
            w.close()
            manifest["shards"][name] = len(chunk)

    pd.DataFrame(rows).to_csv(out / "FINALIST_LIGAND_TABLE.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "FINALIST_PREP_FAILURES.csv", index=False)
    if stereo_changed:
        pd.DataFrame(stereo_changed).to_csv(out / "FINALIST_STEREO_CHANGED.csv", index=False)
    (out / "FINALIST_PREP_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2))
    print(f"shards written: {len(manifest['shards'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
