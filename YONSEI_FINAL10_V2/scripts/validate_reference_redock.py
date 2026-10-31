#!/usr/bin/env python3
"""Validate GNINA reference redocking and enforce the V2 candidate gate.

The repository contains old GNINA redock SDFs, but no recorded command/config
pair for those files.  They are therefore reported as historical diagnostics.
They cannot substitute for a GNINA run using the corrected ligand-centred
configs generated in 01_REFERENCE_POCKETS.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import warnings
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import prolif as plf
from spyrmsd import molecule as spyrmsd_molecule
from spyrmsd.rmsd import rmsdwrapper


SYSTEMS = {
    "5NDD_AZ8838": {
        "reference_ligand": "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/ligands/5NDD_AZ8838_crystallographic_clean.sdf",
        "receptor": "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/receptors/5NDD_inactive_orthosteric_receptor_clean.pdb",
        "corrected_config": "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/configs/BoxA_orthosteric_5NDD_ligand_centered.conf",
        "historical_output": "BIN/GNINA_RUN_EX32_CPU/GNINA_OUT_BoxA_Redock.sdf",
    },
    "5NDZ_AZ3451": {
        "reference_ligand": "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/ligands/5NDZ_AZ3451_crystallographic_clean.sdf",
        "receptor": "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/receptors/5NDZ_inactive_allosteric_receptor_clean.pdb",
        "corrected_config": "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/configs/BoxB_allosteric_5NDZ_ligand_centered.conf",
        "historical_output": "BIN/GNINA_RUN_EX32_CPU/GNINA_OUT_BoxB_Redock.sdf",
    },
}


def residue_string(residue) -> str:
    return f"{residue.name}:{residue.number}:{residue.chain or '-'}"


def pose_fingerprint(pose, receptor) -> set[str]:
    fingerprint = plf.Fingerprint()
    ifp = fingerprint.generate(
        plf.Molecule.from_rdkit(pose, resname="LIG", resnumber=1, chain="A"),
        plf.Molecule.from_rdkit(receptor),
        metadata=True,
    )
    keys = set()
    for pair, interactions in ifp.items():
        protein_residue = residue_string(pair[1])
        for interaction_type, metadata_values in interactions.items():
            for _ in metadata_values:
                keys.add(f"{protein_residue}|{interaction_type}")
    return keys


def prop(mol, name: str) -> str:
    return mol.GetProp(name) if mol is not None and mol.HasProp(name) else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gnina", default=shutil.which("gnina"))
    parser.add_argument("--corrected-output-dir", type=Path, default=None,
                        help="Directory containing outputs from run_reference_redock.py")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "YONSEI_FINAL10_V2/02_REFERENCE_VALIDATION"
    out.mkdir(parents=True, exist_ok=True)
    corrected_output_dir = (args.corrected_output_dir or (root / "YONSEI_FINAL10_V2/01_BENCHMARK_REPAIR/corrected_redock")).resolve()
    reference_summary = json.loads((out / "reference_interactions_summary.json").read_text(encoding="utf-8"))
    result_summary = {
        "gnina_executable": args.gnina,
        "gnina_executable_present": bool(args.gnina),
        "corrected_output_dir": str(corrected_output_dir),
        "corrected_config_benchmark_status": "BLOCKED_NO_COMPATIBLE_GNINA_EXECUTABLE" if not args.gnina else "PENDING_RUN",
        "candidate_redocking_gate": "BLOCKED_UNTIL_CORRECTED_REFERENCE_REDOCK_PASSES",
        "systems": {},
    }
    csv_rows = []
    for system_name, system in SYSTEMS.items():
        reference_path = root / system["reference_ligand"]
        receptor_path = root / system["receptor"]
        historical_path = root / system["historical_output"]
        corrected_path = corrected_output_dir / f"{system_name}_corrected_gnina_redock.sdf"
        evaluated_path = corrected_path if corrected_path.exists() and corrected_path.stat().st_size > 0 else historical_path
        evaluated_as_corrected = evaluated_path == corrected_path
        reference = Chem.MolFromMolFile(str(reference_path), removeHs=False, sanitize=False)
        receptor = Chem.MolFromPDBFile(str(receptor_path), removeHs=False, sanitize=False)
        reference_keys = set(reference_summary["systems"][system_name]["reference_fingerprint_keys"])
        poses = []
        if evaluated_path.exists() and evaluated_path.stat().st_size > 0:
            poses = [mol for mol in Chem.SDMolSupplier(str(evaluated_path), removeHs=False, sanitize=False) if mol is not None]
        pose_rows = []
        reference_spyrmsd = spyrmsd_molecule.Molecule.from_rdkit(reference)
        for pose_index, pose in enumerate(poses, start=1):
            try:
                pose_spyrmsd = spyrmsd_molecule.Molecule.from_rdkit(pose)
                rmsd = float(rmsdwrapper(reference_spyrmsd, pose_spyrmsd, symmetry=True, strip=True, center=False, minimize=False)[0])
            except Exception as error:
                rmsd = None
                rmsd_error = f"{type(error).__name__}: {error}"
            else:
                rmsd_error = ""
            try:
                pose_keys = pose_fingerprint(pose, receptor)
                overlap = sorted(reference_keys & pose_keys)
                recovery = len(overlap) / len(reference_keys) if reference_keys else None
                jaccard = len(overlap) / len(reference_keys | pose_keys) if reference_keys | pose_keys else None
                fingerprint_error = ""
            except Exception as error:
                pose_keys = set()
                overlap = []
                recovery = None
                jaccard = None
                fingerprint_error = f"{type(error).__name__}: {error}"
            row = {
                "system": system_name,
                "pose_index": pose_index,
                "provenance": "corrected_config_gnina_output" if evaluated_as_corrected else "historical_BIN_GNINA_output_diagnostic_only",
                "evaluated_output": str(evaluated_path),
                "historical_output": str(historical_path),
                "corrected_config": str(root / system["corrected_config"]),
                "minimizedAffinity": prop(pose, "minimizedAffinity"),
                "CNNscore": prop(pose, "CNNscore"),
                "CNNaffinity": prop(pose, "CNNaffinity"),
                "symmetry_corrected_heavy_atom_rmsd_angstrom": rmsd,
                "rmsd_error": rmsd_error,
                "reference_fingerprint_key_count": len(reference_keys),
                "pose_fingerprint_key_count": len(pose_keys),
                "interaction_fingerprint_overlap_count": len(overlap),
                "interaction_fingerprint_recovery": recovery,
                "interaction_fingerprint_jaccard": jaccard,
                "overlap_keys": ";".join(overlap),
                "fingerprint_error": fingerprint_error,
                "rmsd_under_2A": rmsd is not None and rmsd < 2.0,
                "rmsd_under_1_5A": rmsd is not None and rmsd < 1.5,
            }
            pose_rows.append(row)
            csv_rows.append(row)
        best_rmsd = min((row["symmetry_corrected_heavy_atom_rmsd_angstrom"] for row in pose_rows if row["symmetry_corrected_heavy_atom_rmsd_angstrom"] is not None), default=None)
        best_recovery = max((row["interaction_fingerprint_recovery"] for row in pose_rows if row["interaction_fingerprint_recovery"] is not None), default=None)
        top_ranked = pose_rows[0] if pose_rows else None
        def numeric(row, key):
            try:
                return float(row[key]) if row and row.get(key, "") != "" else None
            except (TypeError, ValueError):
                return None
        best_rmsd_row = min(
            (row for row in pose_rows if row["symmetry_corrected_heavy_atom_rmsd_angstrom"] is not None),
            key=lambda row: row["symmetry_corrected_heavy_atom_rmsd_angstrom"],
            default=None,
        )
        best_cnn_row = max(
            (row for row in pose_rows if numeric(row, "CNNscore") is not None),
            key=lambda row: numeric(row, "CNNscore"),
            default=None,
        )
        best_affinity_row = min(
            (row for row in pose_rows if numeric(row, "minimizedAffinity") is not None),
            key=lambda row: numeric(row, "minimizedAffinity"),
            default=None,
        )
        corrected_passed = evaluated_as_corrected and best_rmsd is not None and best_rmsd <= 2.0
        result_summary["systems"][system_name] = {
            "corrected_config": str(root / system["corrected_config"]),
            "corrected_config_exists": (root / system["corrected_config"]).exists(),
            "historical_output": str(historical_path),
            "historical_output_exists": historical_path.exists(),
            "corrected_output": str(corrected_path),
            "corrected_output_exists": corrected_path.exists() and corrected_path.stat().st_size > 0,
            "evaluated_output": str(evaluated_path),
            "evaluated_as_corrected_config_output": evaluated_as_corrected,
            "evaluated_pose_count": len(pose_rows),
            "evaluated_best_symmetry_corrected_heavy_atom_rmsd_angstrom": best_rmsd,
            "evaluated_best_interaction_fingerprint_recovery": best_recovery,
            "top_ranked_pose_rmsd_angstrom": top_ranked["symmetry_corrected_heavy_atom_rmsd_angstrom"] if top_ranked else None,
            "top_ranked_pose_interaction_fingerprint_recovery": top_ranked["interaction_fingerprint_recovery"] if top_ranked else None,
            "top_ranked_pose_passed_rmsd_2A": bool(top_ranked and top_ranked["symmetry_corrected_heavy_atom_rmsd_angstrom"] is not None and top_ranked["symmetry_corrected_heavy_atom_rmsd_angstrom"] <= 2.0),
            "best_rmsd_pose_index": best_rmsd_row["pose_index"] if best_rmsd_row else None,
            "best_cnnscore_pose_index": best_cnn_row["pose_index"] if best_cnn_row else None,
            "best_cnnscore": numeric(best_cnn_row, "CNNscore"),
            "best_cnnscore_pose_rmsd_angstrom": best_cnn_row["symmetry_corrected_heavy_atom_rmsd_angstrom"] if best_cnn_row else None,
            "best_cnnscore_pose_interaction_fingerprint_recovery": best_cnn_row["interaction_fingerprint_recovery"] if best_cnn_row else None,
            "best_minimized_affinity_pose_index": best_affinity_row["pose_index"] if best_affinity_row else None,
            "best_minimized_affinity": numeric(best_affinity_row, "minimizedAffinity"),
            "best_minimized_affinity_pose_rmsd_angstrom": best_affinity_row["symmetry_corrected_heavy_atom_rmsd_angstrom"] if best_affinity_row else None,
            "best_minimized_affinity_pose_interaction_fingerprint_recovery": best_affinity_row["interaction_fingerprint_recovery"] if best_affinity_row else None,
            "historical_diagnostic_rmsd_target_met": (not evaluated_as_corrected) and best_rmsd is not None and best_rmsd < 2.0,
            "historical_diagnostic_strong_rmsd_target_met": (not evaluated_as_corrected) and best_rmsd is not None and best_rmsd < 1.5,
            "corrected_config_redock_completed": evaluated_as_corrected,
            "corrected_config_benchmark_passed": corrected_passed,
            "status": "CORRECTED_CONFIG_EVALUATED" if evaluated_as_corrected else "HISTORICAL_DIAGNOSTIC_ONLY_CORRECTED_CONFIG_NOT_RUN",
            "poses": pose_rows,
        }

    corrected_runs = all(system_result["corrected_config_redock_completed"] for system_result in result_summary["systems"].values())
    corrected_pass = corrected_runs and all(system_result["corrected_config_benchmark_passed"] for system_result in result_summary["systems"].values())
    if corrected_pass:
        result_summary["corrected_config_benchmark_status"] = "PASS_BOTH_SYSTEMS_RMSD_UNDER_2A"
        result_summary["candidate_redocking_gate"] = "OPEN_REFERENCE_BENCHMARK_PASSED"
    elif corrected_runs:
        result_summary["corrected_config_benchmark_status"] = "FAIL_REFERENCE_RMSD_TARGET"
    elif not args.gnina:
        result_summary["corrected_config_benchmark_status"] = "BLOCKED_NO_COMPATIBLE_GNINA_EXECUTABLE"

    if csv_rows:
        with (out / "reference_redock_pose_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    (out / "reference_redock_validation.json").write_text(json.dumps(result_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Reference redocking validation",
        "",
        "## Gate status",
        "",
        f"- GNINA executable detected: `{result_summary['gnina_executable_present']}`.",
        f"- Corrected-config benchmark status: **{result_summary['corrected_config_benchmark_status']}**.",
        f"- Candidate redocking gate: **{result_summary['candidate_redocking_gate']}**.",
        "",
        "The repository's `BIN/GNINA_RUN_EX32_CPU/GNINA_OUT_*_Redock.sdf` files are evaluated below as historical diagnostics only unless corrected outputs are present in `corrected_redock/`. Their original command/config provenance is not present, so they cannot establish that the corrected ligand-centred boxes were used.",
        "",
        "| System | Evaluated poses | Best RMSD (Å) | Top-ranked RMSD (Å) | Top-ranked ProLIF recovery | Best ProLIF recovery | Corrected benchmark passed |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for system_name, system_result in result_summary["systems"].items():
        lines.append(f"| {system_name} | {system_result['evaluated_pose_count']} | {system_result['evaluated_best_symmetry_corrected_heavy_atom_rmsd_angstrom']} | {system_result['top_ranked_pose_rmsd_angstrom']} | {system_result['top_ranked_pose_interaction_fingerprint_recovery']} | {system_result['evaluated_best_interaction_fingerprint_recovery']} | {system_result['corrected_config_benchmark_passed']} |")
    lines += [
        "",
        "V2 candidate redocking and V2 rank files are produced only after this script reports `OPEN_REFERENCE_BENCHMARK_PASSED`.",
    ]
    (out / "REFERENCE_REDOCK_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
