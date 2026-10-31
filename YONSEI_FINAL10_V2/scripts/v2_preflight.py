#!/usr/bin/env python3
"""Fail-closed pre-flight validation before any real de novo GNINA screen.

Every gate is checked explicitly and reported. The script exits non-zero if any
GLOBAL gate fails, so it can be wired in front of the screen runner. Individual
bad ligands are listed, not silently dropped.

Gates implemented here (Phases 3, 4, 7, 8):
  P3  receptor and config provenance; parsed box values match the corrected
      ligand-centred definitions exactly
  P4  geometric frame: crystal ligand centroid vs box centre, ligand fully
      inside the box with per-face margins, receptor/ligand frames aligned,
      box contains real pocket geometry, expected pocket residues present,
      no crystallographic ligand left inside the docking receptor
  P7  ligand-preparation integrity: full accounting of every input design and
      a before/after structural comparison on a sample
  P8  family routing audit recomputed independently of the manifest

Nothing here docks anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

EXPECTED_BOX = {
    "5NDD_orthosteric": dict(center_x=16.974, center_y=5.926, center_z=55.826,
                             size_x=17.332, size_y=17.672, size_z=13.644),
    "5NDZ_allosteric": dict(center_x=1.356, center_y=12.419, center_z=48.400,
                            size_x=20.702, size_y=17.885, size_z=25.525),
}
POCKET_FILES = {
    "5NDD_orthosteric": dict(
        config="01_REFERENCE_POCKETS/configs/BoxA_orthosteric_5NDD_ligand_centered.conf",
        receptor_pdbqt="01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdbqt",
        receptor_pdb="01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdb",
        ligand="01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        ligand_code="8TZ",
    ),
    "5NDZ_allosteric": dict(
        config="01_REFERENCE_POCKETS/configs/BoxB_allosteric_5NDZ_ligand_centered.conf",
        receptor_pdbqt="01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdbqt",
        receptor_pdb="01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdb",
        ligand="01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        ligand_code="8UN",
    ),
}
EXPECTED_RESIDUES = {
    "5NDD_orthosteric": ["TYR:82", "LYS:131", "HIS:135", "ILE:152", "PHE:155",
                         "HIS:227", "ASP:228", "LEU:330"],
    "5NDZ_allosteric": ["ALA:120", "LEU:123", "PHE:154", "GLY:157", "ASN:158",
                        "CYS:161", "TRP:199", "ILE:202", "LEU:203", "TYR:210", "VAL:211"],
}
BOX_TOLERANCE = 0.01
FAMILY_POCKETS = {
    "orthosteric": {"5NDD_orthosteric"},
    "allosteric": {"5NDZ_allosteric"},
    "exploratory": {"5NDD_orthosteric", "5NDZ_allosteric"},
}

RESULTS: list[dict] = []


def gate(name: str, passed: bool, detail, fatal: bool = True) -> bool:
    RESULTS.append(dict(gate=name, passed=bool(passed), fatal=fatal, detail=detail))
    mark = "PASS" if passed else ("FAIL" if fatal else "WARN")
    print(f"[{mark}] {name}: {detail}", flush=True)
    return bool(passed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_config(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        try:
            values[key] = float(value.strip())
        except ValueError:
            values[key] = value.strip()
    return values


def receptor_atoms(pdb_path: Path):
    """Heavy-atom coordinates, residue labels and HETATM names from a PDB."""
    coords, labels, het = [], [], Counter()
    for line in pdb_path.read_text().splitlines():
        record = line[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        resname = line[17:20].strip()
        if record == "HETATM":
            het[resname] += 1
        if element.upper() == "H":
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        coords.append(xyz)
        labels.append(f"{resname}:{line[22:26].strip()}:{line[21].strip() or '-'}")
    return np.asarray(coords, dtype=float), labels, het


def phase3_and_4(v2: Path) -> None:
    for pocket, files in POCKET_FILES.items():
        config_path = v2 / files["config"]
        pdb_path = v2 / files["receptor_pdb"]
        pdbqt_path = v2 / files["receptor_pdbqt"]
        ligand_path = v2 / files["ligand"]

        for label, path in (("config", config_path), ("receptor_pdbqt", pdbqt_path),
                            ("receptor_pdb", pdb_path), ("ligand", ligand_path)):
            gate(f"P3 {pocket} {label} exists", path.exists(), str(path))
            if path.exists():
                print(f"       sha256 {sha256(path)}  {path}", flush=True)

        parsed = parse_config(config_path)
        expected = EXPECTED_BOX[pocket]
        deltas = {k: abs(parsed.get(k, float("nan")) - v) for k, v in expected.items()}
        worst = max(deltas.values())
        gate(f"P3 {pocket} box matches corrected definition",
             worst <= BOX_TOLERANCE,
             f"max |delta| = {worst:.6f} A over {len(expected)} fields; "
             f"parsed = { {k: parsed.get(k) for k in expected} }")

        ligand = Chem.MolFromMolFile(str(ligand_path), removeHs=True, sanitize=False)
        gate(f"P4 {pocket} crystal ligand parses", ligand is not None, str(ligand_path))
        if ligand is None:
            continue
        lig_xyz = ligand.GetConformer().GetPositions()
        centroid = lig_xyz.mean(axis=0)
        centre = np.array([expected["center_x"], expected["center_y"], expected["center_z"]])
        offset = float(np.linalg.norm(centroid - centre))
        gate(f"P4 {pocket} ligand centroid at box centre", offset <= 0.5,
             f"centroid = ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), "
             f"offset = {offset:.4f} A")

        half = np.array([expected["size_x"], expected["size_y"], expected["size_z"]]) / 2.0
        lower, upper = centre - half, centre + half
        inside = np.all((lig_xyz >= lower) & (lig_xyz <= upper), axis=1)
        margins = np.minimum(lig_xyz - lower, upper - lig_xyz)
        gate(f"P4 {pocket} all crystal ligand atoms inside box", bool(inside.all()),
             f"{int(inside.sum())}/{len(inside)} heavy atoms inside; "
             f"min margin to any face = {margins.min():.3f} A "
             f"(per-axis min: x {margins[:, 0].min():.3f}, y {margins[:, 1].min():.3f}, "
             f"z {margins[:, 2].min():.3f})")

        rec_xyz, rec_labels, het = receptor_atoms(pdb_path)
        gate(f"P4 {pocket} receptor heavy atoms loaded", len(rec_xyz) > 1000, f"{len(rec_xyz)} atoms")

        distances = np.linalg.norm(rec_xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
        near5 = int((distances.min(axis=1) <= 5.0).sum())
        contacts = int(((distances >= 2.0) & (distances <= 5.0)).sum())
        closest = float(distances.min())
        gate(f"P4 {pocket} receptor/ligand frames aligned", near5 >= 50 and closest <= 5.0,
             f"{near5} receptor heavy atoms within 5 A of the crystal ligand; "
             f"{contacts} atom pairs in 2-5 A; closest contact {closest:.3f} A")

        in_box = np.all((rec_xyz >= lower) & (rec_xyz <= upper), axis=1)
        gate(f"P4 {pocket} box contains pocket geometry", int(in_box.sum()) >= 100,
             f"{int(in_box.sum())} receptor heavy atoms inside the box "
             f"(box volume {np.prod(2 * half):.0f} A^3)")

        near_labels = {rec_labels[i].rsplit(":", 1)[0]
                       for i in np.where(distances.min(axis=1) <= 5.0)[0]}
        expected_res = EXPECTED_RESIDUES[pocket]
        found = [r for r in expected_res if r in near_labels]
        gate(f"P4 {pocket} expected pocket residues present",
             len(found) >= max(3, len(expected_res) // 2),
             f"{len(found)}/{len(expected_res)} within 5 A: found {found}; "
             f"missing {[r for r in expected_res if r not in near_labels]}", fatal=False)

        contaminant = files["ligand_code"]
        gate(f"P4 {pocket} docking receptor free of its crystal ligand",
             het.get(contaminant, 0) == 0,
             f"HETATM residues in receptor: {dict(het) if het else 'none'}; "
             f"looking for {contaminant}")


def phase7(v2: Path, library: Path, queue: Path) -> None:
    table = pd.read_csv(library / "V2_LIGAND_TABLE.csv")
    failures = pd.read_csv(library / "V2_PREP_FAILURES.csv")
    manifest = json.loads((library / "V2_PREP_MANIFEST.json").read_text())
    source = pd.read_csv(queue)

    n_in = len(source)
    n_ok = int(manifest["prepared_ok"])
    n_fail = int(manifest["prepared_failed"])
    n_dupe = int(manifest["duplicate_inchikeys_collapsed"])
    n_unique = int(manifest["unique_ligands"])
    gate("P7 input accounting balances",
         n_ok + n_fail == n_in and n_ok - n_dupe == n_unique,
         f"{n_in} inputs = {n_ok} prepared + {n_fail} failed; "
         f"{n_ok} - {n_dupe} duplicate InChIKeys = {n_unique} unique ligands")

    reasons = failures.prep_reason.value_counts().to_dict() if len(failures) else {}
    gate("P7 every failed design is itemised", len(failures) == n_fail,
         f"{len(failures)} rows listed, reasons: {reasons}")
    if len(failures):
        print("       failed designs:", flush=True)
        for row in failures.itertuples():
            print(f"         {row.cand_id}  {row.prep_reason}  {row.smiles_input}", flush=True)

    # Derive the id prefix from the table itself. A hard-coded prefix silently
    # matched nothing for batches that use a different one, and the gate then
    # "passed" having compared zero molecules.
    prefixes = {str(c)[:2] for c in table.cand_id.head(50)}
    prefix = prefixes.pop() if len(prefixes) == 1 else "DN"
    rank_col = "dock_priority_rank" if "dock_priority_rank" in source.columns else source.columns[0]
    smiles_in = dict(zip((f"{prefix}%05d" % r for r in source[rank_col].astype(int)),
                         source.smiles))
    sample = table.sample(n=min(25, len(table)), random_state=7)
    problems = []
    checked = 0
    for row in sample.itertuples():
        before_smiles = smiles_in.get(row.cand_id)
        if before_smiles is None:
            continue
        before = Chem.MolFromSmiles(before_smiles)
        after = Chem.MolFromSmiles(row.smiles_prepared)
        if before is None or after is None:
            problems.append((row.cand_id, "unparsable"))
            continue
        checked += 1
        b_parent = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(before))
        issues = []
        if b_parent.GetNumHeavyAtoms() != after.GetNumHeavyAtoms():
            issues.append(f"heavy {b_parent.GetNumHeavyAtoms()}->{after.GetNumHeavyAtoms()}")
        if rdMolDescriptors.CalcNumRings(b_parent) != rdMolDescriptors.CalcNumRings(after):
            issues.append("ring count changed")
        if len(Chem.GetMolFrags(after)) != 1:
            issues.append(f"{len(Chem.GetMolFrags(after))} fragments")
        # Heavy-atom skeleton must be identical; only H/charge may move.
        strip = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(after)), isomericSmiles=False)
        if Chem.MolToSmiles(b_parent, isomericSmiles=False) != strip:
            b_formula = rdMolDescriptors.CalcMolFormula(b_parent).rstrip("+-")
            a_formula = rdMolDescriptors.CalcMolFormula(after).rstrip("+-")
            if b_formula.split("H")[0] != a_formula.split("H")[0]:
                issues.append(f"heavy-atom formula {b_formula} -> {a_formula}")
        if issues:
            problems.append((row.cand_id, "; ".join(issues)))
    gate("P7 preparation preserves molecular skeleton",
         not problems and checked >= 20,
         f"{checked} sampled designs compared before/after (id prefix {prefix!r}); "
         f"{len(problems)} problems: {problems[:5]}"
         + ("" if checked >= 20 else "  -- TOO FEW COMPARED: the sample matched no "
                                     "input rows, so this gate proved nothing"))

    bad_geometry = []
    for shard in sorted(p for p in (library / "shards").glob("*.sdf")
                        if not p.name.startswith(".")):
        for mol in Chem.SDMolSupplier(str(shard), removeHs=False, sanitize=False):
            if mol is None:
                bad_geometry.append(("<unparsable>", "SDF record failed to parse"))
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "?"
            if mol.GetNumConformers() != 1:
                bad_geometry.append((name, f"{mol.GetNumConformers()} conformers"))
                continue
            xyz = mol.GetConformer().GetPositions()
            if not np.isfinite(xyz).all():
                bad_geometry.append((name, "non-finite coordinates"))
                continue
            if float(np.ptp(xyz, axis=0).max()) < 1.0:
                bad_geometry.append((name, "collapsed geometry"))
                continue
            bonds = [np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()])
                     for b in mol.GetBonds()]
            if bonds and (min(bonds) < 0.7 or max(bonds) > 2.6):
                bad_geometry.append((name, f"bond length {min(bonds):.2f}-{max(bonds):.2f} A"))
    gate("P7 prepared 3D geometry is valid", not bad_geometry,
         f"{len(bad_geometry)} geometry problems: {bad_geometry[:5]}")


def phase8(library: Path, screen_set: Path) -> None:
    ligands = pd.read_csv(screen_set / "SCREEN_SET_LIGANDS.csv")
    counts = ligands.denovo_family.value_counts().to_dict()
    n_orth = counts.get("orthosteric", 0)
    n_allo = counts.get("allosteric", 0)
    n_expl = counts.get("exploratory", 0)
    predicted = n_orth + n_allo + 2 * n_expl

    routed = defaultdict(set)
    for shard in sorted(p for p in (screen_set / "shards").glob("*.sdf")
                        if not p.name.startswith(".")):
        pocket = shard.name.split("_shard")[0]
        for record in shard.read_text().split("$$$$\n"):
            if record.strip():
                routed[record.lstrip("\n").split("\n", 1)[0].strip()].add(pocket)
    actual_jobs = sum(len(v) for v in routed.values())

    gate("P8 job count equals N_orth + N_allo + 2*N_expl",
         predicted == actual_jobs,
         f"orthosteric {n_orth} + allosteric {n_allo} + 2 x exploratory {n_expl} "
         f"= {predicted}; shards contain {actual_jobs} jobs over {len(routed)} ligands")

    family = dict(zip(ligands.cand_id, ligands.denovo_family))
    misrouted = []
    for cand_id, pockets in routed.items():
        want = FAMILY_POCKETS.get(family.get(cand_id, ""), set())
        if pockets != want:
            misrouted.append((cand_id, family.get(cand_id), sorted(pockets), sorted(want)))
    gate("P8 every ligand routed to exactly its intended pocket(s)", not misrouted,
         f"{len(routed)} ligands audited; {len(misrouted)} misrouted: {misrouted[:5]}")

    unknown = [c for c in routed if c not in family]
    gate("P8 no shard ligand is missing from the screen-set table", not unknown,
         f"{len(unknown)} unknown ids: {unknown[:5]}")


SCORE_FILES = {"V2_LIGAND_POCKET_SCORES.csv", "V2_POSE_SCORES.csv",
               "V2_POSE_INTERACTIONS_SUMMARY.csv"}


def phase0(real_root: Path, dryrun_root: Path, run_out: Path | None) -> None:
    gate("P0 real and dry-run output roots are separate",
         real_root.resolve() != dryrun_root.resolve()
         and dryrun_root.resolve() not in real_root.resolve().parents
         and real_root.resolve() not in dryrun_root.resolve().parents,
         f"real = {real_root}, dryrun = {dryrun_root}")

    # Synthetic/dry-run material must never sit inside the real root.
    contamination = [p for p in real_root.rglob("*") if p.is_file()
                     and p.name in SCORE_FILES
                     and dryrun_root.resolve() in p.resolve().parents]
    gate("P0 no dry-run score files inside the real root", not contamination,
         f"{len(contamination)} contaminating files: {[str(p) for p in contamination[:5]]}")

    # Completed sibling runs are legitimate history, not contamination. What must
    # be clean is the directory THIS run will write into.
    completed = sorted({p.parent.name for p in real_root.rglob("*")
                        if p.is_file() and p.name in SCORE_FILES})
    if run_out is not None:
        existing = [p for p in run_out.rglob("*") if p.is_file() and p.name in SCORE_FILES] \
            if run_out.exists() else []
        gate("P0 this run's output directory holds no previous scores", not existing,
             f"target = {run_out}; {len(existing)} pre-existing score files; "
             f"completed sibling runs (kept, not overwritten): {completed}")
    else:
        gate("P0 completed sibling runs recorded", True,
             f"completed runs under the real root: {completed}", fatal=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--screen-set", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--dryrun-root", type=Path, required=True)
    parser.add_argument("--run-out", type=Path, default=None,
                        help="directory this screen will write into; must hold no prior scores")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    v2 = root / "YONSEI_FINAL10_V2"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 0 — output separation")
    print("=" * 72)
    phase0(args.real_root.resolve(), args.dryrun_root.resolve(),
           args.run_out.resolve() if args.run_out else None)
    print("=" * 72)
    print("PHASE 3/4 — receptor, box and geometric frame")
    print("=" * 72)
    phase3_and_4(v2)
    print("=" * 72)
    print("PHASE 7 — ligand preparation integrity")
    print("=" * 72)
    phase7(v2, args.library.resolve(), args.queue.resolve())
    print("=" * 72)
    print("PHASE 8 — family routing audit")
    print("=" * 72)
    phase8(args.library.resolve(), args.screen_set.resolve())

    fatal_failures = [r for r in RESULTS if not r["passed"] and r["fatal"]]
    warnings = [r for r in RESULTS if not r["passed"] and not r["fatal"]]
    verdict = "PASS" if not fatal_failures else "FAIL"
    report = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        verdict=verdict,
        gates_total=len(RESULTS),
        gates_passed=sum(1 for r in RESULTS if r["passed"]),
        fatal_failures=[r["gate"] for r in fatal_failures],
        warnings=[r["gate"] for r in warnings],
        results=RESULTS,
    )
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print("=" * 72)
    print(f"PRE-FLIGHT VERDICT: {verdict}  "
          f"({report['gates_passed']}/{report['gates_total']} gates passed, "
          f"{len(warnings)} warnings)")
    if fatal_failures:
        for r in fatal_failures:
            print(f"  FATAL: {r['gate']} — {r['detail']}")
    print("=" * 72)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
