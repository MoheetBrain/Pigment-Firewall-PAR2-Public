#!/usr/bin/env python3
"""ProLIF interaction fingerprints for V2 docked poses.

For each ligand-pocket pair, fingerprints the requested pose against the
corrected receptor and compares it with the crystallographic antagonist
fingerprint for that pocket (AZ8838 in 5NDD, AZ3451 in 5NDZ).

Recovery is the fraction of crystal-reference residue|interaction keys that the
docked pose reproduces. It is a geometric similarity measure, not evidence of
binding.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import prolif as plf  # noqa: E402

# Protonated receptors and repaired reference fingerprints, produced by
# v2_reference_ifp_repair.py. The unprotonated receptors cannot support
# hydrogen-bond detection at all, so they must not be used here.
POCKET_RECEPTOR = {
    "5NDD_orthosteric": "02_REFERENCE_VALIDATION_REPAIRED/5NDD_orthosteric_receptor_protonated_pH74.pdb",
    "5NDZ_allosteric": "02_REFERENCE_VALIDATION_REPAIRED/5NDZ_allosteric_receptor_protonated_pH74.pdb",
}
POCKET_REFERENCE = {"5NDD_orthosteric": "5NDD_AZ8838", "5NDZ_allosteric": "5NDZ_AZ3451"}
REPAIRED_REFERENCE = "02_REFERENCE_VALIDATION_REPAIRED/REFERENCE_IFP_REPAIRED.json"
HBOND_TYPES = {"HBDonor", "HBAcceptor"}

_RECEPTORS: dict[str, object] = {}


def load_receptor(root: Path, pocket: str):
    if pocket not in _RECEPTORS:
        path = root / "YONSEI_FINAL10_V2" / POCKET_RECEPTOR[pocket]
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if mol is None:
            raise RuntimeError(f"could not read receptor {path}")
        _RECEPTORS[pocket] = plf.Molecule.from_rdkit(mol)
    return _RECEPTORS[pocket]


def reference_keys(root: Path) -> dict[str, set[str]]:
    """Repaired crystallographic reference keys (protonated, corrected tautomer)."""
    path = root / "YONSEI_FINAL10_V2" / REPAIRED_REFERENCE
    if not path.exists():
        raise SystemExit(
            f"repaired reference fingerprints not found at {path}. "
            "Run v2_reference_ifp_repair.py first; the unrepaired reference cannot "
            "detect hydrogen bonds and must not be used.")
    report = json.loads(path.read_text())
    if report.get("verdict") != "PASS":
        raise SystemExit(f"reference IFP repair did not pass: {report.get('failures')}")
    return {pocket: set(report["systems"][name]["reference_fingerprint_keys"])
            for pocket, name in POCKET_REFERENCE.items()}


def pose_keys(ligand_mol: Chem.Mol, receptor) -> tuple[set[str], int, int]:
    ligand = plf.Molecule.from_rdkit(ligand_mol, resname="LIG", resnumber=1, chain="A")
    fingerprint = plf.Fingerprint()
    ifp = fingerprint.generate(ligand, receptor, metadata=True)
    keys: set[str] = set()
    total = 0
    hbonds = 0
    for pair, interactions in ifp.items():
        protein_residue = pair[1]
        residue = f"{protein_residue.name}:{protein_residue.number}:{protein_residue.chain or '-'}"
        for interaction_name in interactions:
            keys.add(f"{residue}|{interaction_name}")
            total += 1
            if interaction_name in HBOND_TYPES:
                hbonds += 1
    return keys, total, hbonds


def process_file(job: tuple) -> list[dict]:
    path_str, pocket, root_str, which_pose, max_pose, wanted = job
    root = Path(root_str)
    receptor = load_receptor(root, pocket)
    refs = reference_keys(root)[pocket]
    rows: list[dict] = []
    seen: dict[str, int] = defaultdict(int)
    opener = gzip.open if path_str.endswith(".gz") else open
    with opener(path_str, "rb") as handle:
        for mol in Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=True):
            if mol is None:
                continue
            cand_id = mol.GetProp("cand_id") if mol.HasProp("cand_id") else (
                mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            if not cand_id:
                continue
            # Pre-filter: skip molecules outside the requested set before doing
            # any fingerprint work, so a shortlist run is genuinely cheap.
            if wanted is not None and cand_id not in wanted:
                continue
            seen[cand_id] += 1
            rank = seen[cand_id]
            if which_pose == "top" and rank != 1:
                continue
            if rank > max_pose:
                continue
            try:
                keys, total, hbonds = pose_keys(Chem.AddHs(mol, addCoords=True), receptor)
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(cand_id=cand_id, pocket=pocket, pose_rank=rank,
                                 ifp_recovery=None, ifp_n_interactions=None,
                                 ifp_hbonds=None,
                                 ifp_status=f"FAILED:{type(exc).__name__}", ifp_keys=""))
                continue
            recovery = len(keys & refs) / len(refs) if refs else None
            rows.append(dict(cand_id=cand_id, pocket=pocket, pose_rank=rank,
                             ifp_recovery=recovery, ifp_n_interactions=total,
                             ifp_hbonds=hbonds,
                             ifp_shared_keys=len(keys & refs), ifp_reference_keys=len(refs),
                             ifp_status="OK", ifp_keys=";".join(sorted(keys))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pose", choices=["top", "all"], default="top")
    parser.add_argument("--max-pose", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-cand-ids", type=Path, default=None,
                        help="optional CSV with a cand_id column limiting the analysis")
    args = parser.parse_args()

    root = args.root.resolve()
    screen = args.screen.resolve()
    out = (args.out or screen).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(p for p in (screen / "poses").glob("*_out.sdf*")
                        if not p.name.startswith("."))
    if not pose_files:
        raise SystemExit(f"no pose files under {screen / 'poses'}")
    wanted = None
    if args.only_cand_ids and args.only_cand_ids.exists():
        wanted = set(pd.read_csv(args.only_cand_ids).cand_id)
        print(f"[ifp] restricting to {len(wanted)} cand_ids", flush=True)
    jobs = [(str(p), p.name.split("_shard")[0], str(root), args.pose, args.max_pose, wanted)
            for p in pose_files]

    rows: list[dict] = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for n, chunk in enumerate(pool.imap_unordered(process_file, jobs), 1):
                rows.extend(chunk)
                print(f"[ifp] {n}/{len(jobs)} pose files", flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            rows.extend(process_file(job))
            print(f"[ifp] {n}/{len(jobs)} pose files", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "V2_POSE_INTERACTIONS.csv", index=False)

    best = (frame[frame.ifp_recovery.notna()]
            .sort_values("ifp_recovery", ascending=False)
            .drop_duplicates(["cand_id", "pocket"]))
    best = best.rename(columns={"pose_rank": "ifp_best_pose_rank"})
    best[["cand_id", "pocket", "ifp_best_pose_rank", "ifp_recovery", "ifp_n_interactions",
          "ifp_hbonds", "ifp_shared_keys", "ifp_reference_keys", "ifp_keys"]].to_csv(
        out / "V2_POSE_INTERACTIONS_SUMMARY.csv", index=False)

    summary = dict(
        prolif_version=plf.__version__,
        pose_files=len(pose_files),
        rows=int(len(frame)),
        ligand_pocket_pairs=int(len(best)),
        failures=int((frame.ifp_status != "OK").sum()),
        recovery_by_pocket={
            pocket: dict(median=float(sub.ifp_recovery.median()),
                         max=float(sub.ifp_recovery.max()),
                         mean_hbonds=float(sub.ifp_hbonds.mean()),
                         poses_with_hbond=int((sub.ifp_hbonds > 0).sum()))
            for pocket, sub in best.groupby("pocket")
        },
        reference_source="02_REFERENCE_VALIDATION_REPAIRED (protonated, corrected tautomer)",
        note="Fingerprint recovery is geometric similarity to the crystal antagonist, not activity.",
    )
    (out / "V2_INTERACTIONS_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
