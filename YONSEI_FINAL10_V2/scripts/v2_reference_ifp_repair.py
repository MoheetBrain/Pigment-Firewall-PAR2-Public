#!/usr/bin/env python3
"""Rebuild the crystallographic reference interaction fingerprints correctly.

Repairs the two defects recorded in 04_VERIFICATION/REFERENCE_MODEL_DEFECTS.md:

  1. the stored 5NDD ligand is a non-aromatic 4H-imidazole rather than AZ8838.
     Bond orders are reassigned from the wwPDB template SMILES onto the
     crystallographic heavy-atom coordinates, so the geometry is untouched and
     only the chemistry is corrected.
  2. neither ligand nor receptor carried explicit hydrogens, so ProLIF's
     donor/acceptor detectors could never fire. Both are now protonated: the
     ligand with RDKit (coordinates generated for the added hydrogens) and the
     receptor with Open Babel at pH 7.4.

Nothing is forced. No interaction is asserted by hand; every term still comes
from ProLIF's own geometric criteria. The script runs a four-condition control
so the effect of each repair is separable and auditable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import prolif as plf  # noqa: E402

# wwPDB chemical-component ground truth.
SYSTEMS = {
    "5NDD_AZ8838": dict(
        pocket="5NDD_orthosteric",
        ligand="01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        receptor="01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdb",
        template_smiles="CCCc1cc(F)ccc1[C@H](O)c1[nH]ccn1",
        expected_inchikey="IDFPQEHZYBXIFO-LBPRGKRZSA-N",
        ligand_code="8TZ",
    ),
    "5NDZ_AZ3451": dict(
        pocket="5NDZ_allosteric",
        ligand="01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        receptor="01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdb",
        template_smiles=("C[C@@H](C1CCCCC1)n1c(-c2cc3c(cc2Br)OCO3)nc2cc"
                         "(C(=O)Nc3ccc(C#N)cc3)ccc21"),
        expected_inchikey="FJAOGFGHTPYADT-SFHVURJKSA-N",
        ligand_code="8UN",
    ),
}
HBOND_TYPES = {"HBDonor", "HBAcceptor"}

# Heavy-atom donor/acceptor probe used only as a diagnostic: it explains a zero
# hydrogen-bond count instead of assuming one must exist. A genuinely lipophilic
# pocket is allowed to produce no hydrogen bond.
POLAR = Chem.MolFromSmarts("[$([N,O,S;!H0]),$([N,O,S;$([n,o,s]),$([N,O,S;H0])])]")


def protonate_receptor(pdb_in: Path, pdb_out: Path, ph: float = 7.4) -> dict:
    """Add hydrogens to the receptor with Open Babel at the given pH."""
    result = subprocess.run(
        ["obabel", str(pdb_in), "-O", str(pdb_out), "-h", "-p", str(ph)],
        capture_output=True, text=True)
    if not pdb_out.exists() or pdb_out.stat().st_size == 0:
        raise RuntimeError(f"obabel failed to protonate {pdb_in}: {result.stderr[:400]}")
    heavy = hydro = 0
    for line in pdb_out.read_text().splitlines():
        if line[:6] in ("ATOM  ", "HETATM"):
            element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
            if element == "H":
                hydro += 1
            else:
                heavy += 1
    return dict(tool="openbabel", ph=ph, heavy_atoms=heavy, hydrogens_added=hydro,
                stderr_tail=result.stderr.strip()[-200:])


def corrected_ligand(sdf: Path, template_smiles: str) -> tuple[Chem.Mol, dict]:
    """Reassign bond orders from the template, keeping crystal coordinates."""
    raw = Chem.MolFromMolFile(str(sdf), removeHs=False, sanitize=False)
    if raw is None:
        raise RuntimeError(f"cannot read {sdf}")
    Chem.SanitizeMol(raw, Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE, catchErrors=True)
    stored_key = Chem.MolToInchiKey(raw)
    template = Chem.MolFromSmiles(template_smiles)
    fixed = AllChem.AssignBondOrdersFromTemplate(template, raw)
    Chem.AssignStereochemistryFrom3D(fixed)
    before = fixed.GetConformer().GetPositions()
    with_h = Chem.AddHs(fixed, addCoords=True)
    after = with_h.GetConformer().GetPositions()[:len(before)]
    max_shift = float(abs(before - after).max())
    return with_h, dict(
        stored_inchikey=stored_key,
        corrected_inchikey=Chem.MolToInchiKey(fixed),
        aromatic_bonds_stored=sum(b.GetIsAromatic() for b in raw.GetBonds()),
        aromatic_bonds_corrected=sum(b.GetIsAromatic() for b in fixed.GetBonds()),
        heavy_atoms=fixed.GetNumHeavyAtoms(),
        hydrogens_added=with_h.GetNumAtoms() - fixed.GetNumAtoms(),
        max_heavy_atom_coordinate_shift=max_shift,
    )


def fingerprint(ligand: Chem.Mol, receptor_pdb: Path) -> tuple[list[str], Counter, list[dict]]:
    receptor_rd = Chem.MolFromPDBFile(str(receptor_pdb), removeHs=False, sanitize=False)
    if receptor_rd is None:
        raise RuntimeError(f"cannot read receptor {receptor_pdb}")
    receptor = plf.Molecule.from_rdkit(receptor_rd)
    lig = plf.Molecule.from_rdkit(ligand, resname="LIG", resnumber=1, chain="A")
    fp = plf.Fingerprint()
    ifp = fp.generate(lig, receptor, metadata=True)
    keys, counts, detail = [], Counter(), []
    for pair, interactions in ifp.items():
        residue = pair[1]
        label = f"{residue.name}:{residue.number}:{residue.chain or '-'}"
        for name, metadata in interactions.items():
            keys.append(f"{label}|{name}")
            counts[name] += 1
            entry = dict(residue=label, interaction=name)
            meta = metadata[0] if isinstance(metadata, (list, tuple)) and metadata else metadata
            if isinstance(meta, dict):
                for field in ("distance", "DHA_angle", "angle"):
                    if field in meta:
                        entry[field] = round(float(meta[field]), 3)
            detail.append(entry)
    return sorted(set(keys)), counts, detail


def polar_contact_diagnostic(ligand: Chem.Mol, receptor_pdb: Path) -> dict:
    """Why a system reports no hydrogen bond: are polar partners even in range?"""
    import numpy as np
    lig_conf = ligand.GetConformer().GetPositions()
    lig_polar = [i for i in range(ligand.GetNumAtoms())
                 if ligand.GetAtomWithIdx(i).GetSymbol() in ("N", "O", "S")]
    lig_h = sum(1 for a in ligand.GetAtoms() if a.GetSymbol() == "H")
    lig_donor_h = sum(1 for a in ligand.GetAtoms()
                      if a.GetSymbol() == "H" and a.GetDegree() == 1
                      and a.GetNeighbors()[0].GetSymbol() in ("N", "O", "S"))
    rec_polar, rec_h = [], 0
    for line in receptor_pdb.read_text().splitlines():
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if element == "H":
            rec_h += 1
        elif element in ("N", "O", "S"):
            rec_polar.append((xyz, f"{line[17:20].strip()}:{line[22:26].strip()}", line[12:16].strip()))
    if not lig_polar or not rec_polar:
        return dict(ligand_polar_atoms=len(lig_polar), receptor_polar_atoms=len(rec_polar))
    lp = np.array([lig_conf[i] for i in lig_polar])
    rp = np.array([p[0] for p in rec_polar])
    dist = np.linalg.norm(rp[:, None, :] - lp[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0][:5]
    closest = [dict(receptor_atom=f"{rec_polar[int(i)][1]}.{rec_polar[int(i)][2]}",
                    ligand_atom_idx=int(lig_polar[int(j)]),
                    ligand_element=ligand.GetAtomWithIdx(int(lig_polar[int(j)])).GetSymbol(),
                    distance=round(float(dist[int(i), int(j)]), 3)) for i, j in order]
    return dict(ligand_polar_atoms=len(lig_polar), ligand_hydrogens=lig_h,
                ligand_polar_hydrogens=lig_donor_h, receptor_polar_atoms=len(rec_polar),
                receptor_hydrogens=rec_h, min_polar_distance=round(float(dist.min()), 3),
                polar_pairs_within_3_5A=int((dist <= 3.5).sum()),
                closest_polar_contacts=closest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    v2 = root / "YONSEI_FINAL10_V2"
    out = (args.out or v2 / "02_REFERENCE_VALIDATION_REPAIRED").resolve()
    out.mkdir(parents=True, exist_ok=True)

    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  prolif_version=plf.__version__, rdkit_version=Chem.rdBase.rdkitVersion,
                  systems={})
    failures = []

    for name, spec in SYSTEMS.items():
        print("=" * 72)
        print(name)
        print("=" * 72)
        ligand_sdf = v2 / spec["ligand"]
        receptor_pdb = v2 / spec["receptor"]
        receptor_h = out / f"{spec['pocket']}_receptor_protonated_pH74.pdb"

        rec_info = protonate_receptor(receptor_pdb, receptor_h)
        print(f"  receptor protonation: {rec_info['hydrogens_added']} H added to "
              f"{rec_info['heavy_atoms']} heavy atoms at pH {rec_info['ph']}")

        lig_h, lig_info = corrected_ligand(ligand_sdf, spec["template_smiles"])
        ok_key = lig_info["corrected_inchikey"] == spec["expected_inchikey"]
        print(f"  ligand stored InChIKey    : {lig_info['stored_inchikey']}")
        print(f"  ligand corrected InChIKey : {lig_info['corrected_inchikey']} "
              f"({'matches wwPDB' if ok_key else 'DOES NOT MATCH wwPDB'})")
        print(f"  aromatic bonds {lig_info['aromatic_bonds_stored']} -> "
              f"{lig_info['aromatic_bonds_corrected']}; "
              f"{lig_info['hydrogens_added']} H added; "
              f"max heavy-atom coordinate shift {lig_info['max_heavy_atom_coordinate_shift']:.6f} A")
        if not ok_key:
            failures.append(f"{name}: corrected InChIKey does not match wwPDB")
        if lig_info["max_heavy_atom_coordinate_shift"] > 1e-6:
            failures.append(f"{name}: heavy-atom coordinates moved during repair")

        lig_noh = Chem.RemoveHs(Chem.Mol(lig_h))
        stored_noh = Chem.MolFromMolFile(str(ligand_sdf), removeHs=True, sanitize=False)
        conditions = {}
        for label, lig, rec in (
                ("A_as_committed", stored_noh, receptor_pdb),
                ("B_ligandH_storedTautomer", Chem.AddHs(stored_noh, addCoords=True), receptor_pdb),
                ("C_ligandH_correctedTautomer", lig_h, receptor_pdb),
                ("D_repaired_ligandH_receptorH", lig_h, receptor_h)):
            try:
                keys, counts, detail = fingerprint(lig, rec)
            except Exception as exc:  # noqa: BLE001
                conditions[label] = dict(error=f"{type(exc).__name__}: {exc}")
                continue
            hbonds = sum(v for k, v in counts.items() if k in HBOND_TYPES)
            conditions[label] = dict(interactions=len(keys), hydrogen_bonds=hbonds,
                                     types=dict(counts), keys=keys, detail=detail)
            print(f"  {label:32s} interactions {len(keys):3d}  H-bonds {hbonds}")

        final = conditions.get("D_repaired_ligandH_receptorH", {})
        if "error" in final:
            failures.append(f"{name}: repaired condition failed: {final['error']}")
            continue
        # A zero hydrogen-bond count is a finding, not automatically a failure:
        # the 5NDZ allosteric site is a lipophilic pocket. What must be true is
        # that hydrogen-bond DETECTION is functional, i.e. explicit hydrogens
        # exist on both partners and polar atoms are present. Whether ProLIF then
        # fires is left to its own geometric criteria.
        diagnostic = polar_contact_diagnostic(lig_h, receptor_h)
        if diagnostic.get("ligand_polar_hydrogens", 0) == 0 and diagnostic.get("ligand_polar_atoms", 0) > 0:
            failures.append(f"{name}: ligand has polar atoms but no polar hydrogens after repair")
        if diagnostic.get("receptor_hydrogens", 0) == 0:
            failures.append(f"{name}: receptor carries no hydrogens after protonation")
        print(f"  polar diagnostic: ligand {diagnostic.get('ligand_polar_atoms')} polar atoms "
              f"({diagnostic.get('ligand_polar_hydrogens')} polar H), receptor "
              f"{diagnostic.get('receptor_polar_atoms')} polar atoms / "
              f"{diagnostic.get('receptor_hydrogens')} H; closest polar contact "
              f"{diagnostic.get('min_polar_distance')} A; "
              f"{diagnostic.get('polar_pairs_within_3_5A')} polar pairs within 3.5 A")
        if final["hydrogen_bonds"] == 0:
            print("  NOTE: no hydrogen bond detected. Closest polar contacts:")
            for c in diagnostic.get("closest_polar_contacts", [])[:5]:
                print(f"    {c['receptor_atom']:14s} .. ligand {c['ligand_element']}"
                      f"{c['ligand_atom_idx']:<3d} {c['distance']} A")

        control = conditions.get("A_as_committed", {})
        if control.get("hydrogen_bonds", 0) != 0:
            failures.append(f"{name}: control condition unexpectedly found hydrogen bonds")

        print("  hydrogen bonds detected after repair:")
        for entry in final.get("detail", []):
            if entry["interaction"] in HBOND_TYPES:
                print(f"    {entry['residue']:16s} {entry['interaction']:12s} "
                      + "  ".join(f"{k}={v}" for k, v in entry.items()
                                  if k not in ("residue", "interaction")))

        report["systems"][name] = dict(
            pocket=spec["pocket"], ligand_code=spec["ligand_code"],
            receptor_protonation=rec_info, ligand_repair=lig_info,
            expected_inchikey=spec["expected_inchikey"],
            inchikey_matches_wwpdb=ok_key,
            conditions={k: {kk: vv for kk, vv in v.items() if kk != "detail"}
                        for k, v in conditions.items()},
            reference_fingerprint_keys=final.get("keys", []),
            reference_fingerprint_key_count=len(final.get("keys", [])),
            polar_diagnostic=diagnostic,
            hydrogen_bond_detail=[e for e in final.get("detail", [])
                                  if e["interaction"] in HBOND_TYPES],
            protonated_receptor=str(receptor_h),
        )
        writer = Chem.SDWriter(str(out / f"{name}_corrected_protonated_ligand.sdf"))
        writer.write(lig_h)
        writer.close()

    total_hbonds = sum(v["conditions"]["D_repaired_ligandH_receptorH"]["hydrogen_bonds"]
                       for v in report["systems"].values()
                       if "hydrogen_bonds" in v["conditions"].get("D_repaired_ligandH_receptorH", {}))
    if total_hbonds == 0:
        failures.append("hydrogen-bond detection produced no hydrogen bond in either system, "
                        "so the detector cannot be shown to work")
    report["hydrogen_bond_detection_proven"] = total_hbonds > 0
    report["verdict"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    (out / "REFERENCE_IFP_REPAIRED.json").write_text(json.dumps(report, indent=2, default=str))
    print("=" * 72)
    print(f"REFERENCE IFP REPAIR: {report['verdict']}")
    for f in failures:
        print("  FAILURE:", f)
    print("=" * 72)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
