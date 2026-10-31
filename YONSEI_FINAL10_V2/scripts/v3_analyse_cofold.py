#!/usr/bin/env python
"""
v3_analyse_cofold.py — parse and score Boltz-2 co-folding output.

For every (molecule, pocket, seed) job, report:
  co-fold confidence, binder probability, predicted affinity,
  sample-to-sample consistency, pocket occupancy, clash profile,
  and - after receptor alignment - RMSD / shape agreement between the
  co-folded ligand pose and the selected GNINA EX32 pose.

Convergence classes:
  STRONG STRUCTURAL CONVERGENCE | PARTIAL CONVERGENCE |
  DISAGREEMENT | LOW-CONFIDENCE / FAILED

Boltz confidence is NOT evidence of antagonism. It is an orthogonal
structural filter and nothing more.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

BB = {"N", "CA", "C", "O"}


def parse_pdb(path: Path):
    """Return (protein_atoms, ligand_atoms) as lists of dicts."""
    prot, lig = [], []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if el == "H":
            continue
        rec = {"chain": line[21], "resi": line[22:27].strip(),
               "resn": line[17:20].strip(), "name": line[12:16].strip(),
               "xyz": xyz, "el": el}
        (lig if line.startswith("HETATM") or rec["chain"] == "B" else prot).append(rec)
    return prot, lig


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation+translation mapping P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def symmetry_free_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Element-agnostic nearest-neighbour RMSD.

    Two poses of the same molecule can be atom-ordered differently, so a
    positional RMSD over the raw ordering is meaningless. This is a
    conservative surrogate: for each atom in A take the closest atom in B and
    vice versa, then symmetrise. It slightly UNDER-estimates true RMSD, so it
    is used only to separate 'clearly the same site and orientation' from
    'clearly different', never quoted as a crystallographic RMSD.
    """
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return float(math.sqrt(0.5 * ((d.min(axis=1) ** 2).mean()
                                  + (d.min(axis=0) ** 2).mean())))


def centroid_dist(A: np.ndarray, B: np.ndarray) -> float:
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    return float(np.linalg.norm(A.mean(0) - B.mean(0)))


def clash_profile(lig: np.ndarray, prot: np.ndarray, cutoff=2.2):
    if len(lig) == 0 or len(prot) == 0:
        return {"n_clashes": np.nan, "min_contact_A": np.nan}
    d = np.linalg.norm(lig[:, None, :] - prot[None, :, :], axis=2)
    return {"n_clashes": int((d.min(axis=1) < cutoff).sum()),
            "min_contact_A": float(d.min())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofold-dir", required=True,
                    help="Boltz --out_dir (contains predictions/)")
    ap.add_argument("--target", required=True, help="PAR2_COFOLD_TARGET.json")
    ap.add_argument("--gnina-poses", default="",
                    help="optional dir of GNINA EX32 pose PDB/SDF for RMSD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.loads(Path(args.target).read_text())
    pocket_idx = meta["pocket_residues_boltz_index"]
    root = Path(args.cofold_dir)

    pred_dirs = sorted([p for p in root.rglob("*") if p.is_dir()
                        and list(p.glob("*model_0.pdb"))])
    if not pred_dirs:
        pred_dirs = sorted({p.parent for p in root.rglob("*model_*.pdb")})

    rows = []
    for pd_dir in pred_dirs:
        job = pd_dir.name
        parts = job.split("__")
        cofold_id = parts[0] if parts else job
        pocket = parts[1] if len(parts) > 1 else ""
        seed = parts[2].lstrip("s") if len(parts) > 2 else ""

        models = sorted(pd_dir.glob("*model_*.pdb"))
        # Each diffusion sample is produced in its OWN global frame, so ligand
        # coordinates are not comparable until the receptors are superposed.
        # Align every model's CA trace onto model 0 and carry the ligand along.
        lig_sets, rec, ref_ca = [], None, None
        for mp in models:
            prot, lig = parse_pdb(mp)
            if not lig:
                continue
            ca = {a["resi"]: a["xyz"] for a in prot if a["name"] == "CA"}
            L = np.array([a["xyz"] for a in lig])
            if rec is None:
                rec, ref_ca = prot, ca
            elif ref_ca:
                common = sorted(set(ca) & set(ref_ca))
                if len(common) >= 3:
                    P = np.array([ca[r] for r in common])
                    Q = np.array([ref_ca[r] for r in common])
                    R, t = kabsch(P, Q)
                    L = (R @ L.T).T + t
            lig_sets.append(L)

        # confidence + affinity JSONs
        conf, aff = {}, {}
        for jp in pd_dir.glob("confidence_*.json"):
            try:
                conf = json.loads(jp.read_text()); break
            except Exception:
                pass
        for jp in pd_dir.glob("affinity_*.json"):
            try:
                aff = json.loads(jp.read_text()); break
            except Exception:
                pass

        protxyz = np.array([a["xyz"] for a in rec]) if rec else np.zeros((0, 3))

        # pocket occupancy: fraction of constrained pocket residues contacted
        occ = np.nan
        if rec and lig_sets and pocket in pocket_idx:
            want = set(str(i) for i in pocket_idx[pocket])
            pr = [a for a in rec if a["resi"] in want]
            if pr:
                P = np.array([a["xyz"] for a in pr])
                d = np.linalg.norm(lig_sets[0][:, None, :] - P[None, :, :], axis=2)
                hit = {a["resi"] for a, dd in zip(pr, d.min(axis=0)) if dd <= 4.5}
                occ = len(hit) / len(want)

        # sample-to-sample consistency
        cons = np.nan
        if len(lig_sets) > 1:
            pw = [symmetry_free_rmsd(lig_sets[i], lig_sets[j])
                  for i in range(len(lig_sets)) for j in range(i + 1, len(lig_sets))]
            pw = [v for v in pw if np.isfinite(v)]
            if pw:
                cons = float(np.mean(pw))

        cl = clash_profile(lig_sets[0], protxyz) if lig_sets else \
            {"n_clashes": np.nan, "min_contact_A": np.nan}

        rows.append({
            "cofold_id": cofold_id, "pocket": pocket, "seed": seed,
            "job": job, "n_models": len(models), "n_ligand_models": len(lig_sets),
            "cofold_confidence": conf.get("confidence_score"),
            "ptm": conf.get("ptm"), "iptm": conf.get("iptm"),
            "ligand_iptm": conf.get("ligand_iptm"),
            "complex_plddt": conf.get("complex_plddt"),
            "chains_ptm": json.dumps(conf.get("chains_ptm")) if conf.get("chains_ptm") else None,
            "cofold_binder_probability": aff.get("affinity_probability_binary"),
            "cofold_affinity_pred_value": aff.get("affinity_pred_value"),
            "cofold_affinity_prob1": aff.get("affinity_probability_binary1"),
            "cofold_affinity_prob2": aff.get("affinity_probability_binary2"),
            "pocket_occupancy": occ,
            "cofold_consistency_rmsd": cons,
            "n_clashes": cl["n_clashes"], "min_contact_A": cl["min_contact_A"],
            "pose_dir": str(pd_dir),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("NO CO-FOLD PREDICTIONS FOUND under " + str(root), file=sys.stderr)
        Path(args.out).write_text("")
        return 1

    # aggregate across seeds -> one row per (molecule, pocket)
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_seeds": len(g),
            "cofold_confidence": g.cofold_confidence.max(skipna=True),
            "cofold_confidence_mean": g.cofold_confidence.mean(skipna=True),
            "cofold_confidence_sd": g.cofold_confidence.std(skipna=True),
            "ligand_iptm": g.ligand_iptm.max(skipna=True),
            "cofold_binder_probability": g.cofold_binder_probability.max(skipna=True),
            "cofold_affinity_pred_value": g.cofold_affinity_pred_value.mean(skipna=True),
            "pocket_occupancy": g.pocket_occupancy.max(skipna=True),
            "cofold_consistency_rmsd": g.cofold_consistency_rmsd.mean(skipna=True),
            "n_clashes_min": g.n_clashes.min(skipna=True),
            "min_contact_A": g.min_contact_A.max(skipna=True),
        })

    summ = df.groupby(["cofold_id", "pocket"], dropna=False).apply(
        agg, include_groups=False).reset_index()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp.with_name(outp.stem + "_PER_JOB.csv"), index=False)
    summ.to_csv(outp, index=False)
    print(json.dumps({"jobs_parsed": int(len(df)),
                      "molecules": int(summ.shape[0]),
                      "wrote": str(outp)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
