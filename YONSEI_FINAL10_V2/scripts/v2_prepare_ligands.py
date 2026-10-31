#!/usr/bin/env python3
"""Build the V2 dual-pocket docking library.

Merges the de novo docking queue with the sourceable candidate universe,
standardises structures, applies an approximate pH 7.4 protonation model,
embeds a single low-energy 3D conformer per ligand, and writes sharded SDF
files plus a routing manifest for the corrected 5NDD / 5NDZ pockets.

This stage is deterministic given a fixed seed and RDKit version. It performs
no docking and makes no claim about biological activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

POCKET_5NDD = "5NDD_orthosteric"
POCKET_5NDZ = "5NDZ_allosteric"

# Approximate pH 7.4 ionisation rules. These are documented heuristics applied
# by direct atom edits, not a pKa prediction. Anything outside these patterns is
# left exactly as parsed.
ACID_OH = Chem.MolFromSmarts(
    "[$([OX2H1][CX3]=[OX1]),$([OX2H1][SX4](=[OX1])=[OX1]),$([OX2H1][SX3]=[OX1]),$([OX2H1][PX4]=[OX1])]"
)
TETRAZOLE_NH = Chem.MolFromSmarts("[nX3H1]1nnnc1")
AMIDINE_GUANIDINE = Chem.MolFromSmarts(
    "[NX2;H0,H1;!$(N[a]);!$(NC=O);!$(N-[!#6])]=[CX3]([NX3;!$(N[a])])"
)
ALIPHATIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(N[a]);!$(NC=[O,S,N]);!$(NS(=O)=O);!$(NC#N);!$(N[N,O,S]);!$(N=*);!$(N#*)]"
)
MAX_ABS_FORMAL_CHARGE = 2


@dataclass
class PrepResult:
    cand_id: str
    ok: bool
    reason: str
    inchikey: str = ""
    smiles_prepared: str = ""
    formal_charge: int = 0
    heavy_atoms: int = 0
    molblock: str = ""
    mw: float = 0.0
    clogp: float = 0.0
    tpsa: float = 0.0
    hba: int = 0
    hbd: int = 0
    rotb: int = 0
    fsp3: float = 0.0
    qed: float = 0.0
    murcko: str = ""


def _set_ion(mol: Chem.Mol, idx: int, charge: int, delta_h: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0:
        return False
    new_h = atom.GetTotalNumHs() + delta_h
    if new_h < 0:
        return False
    atom.SetFormalCharge(charge)
    atom.SetNumExplicitHs(new_h)
    atom.SetNoImplicit(True)
    return True


def protonate_ph74(mol: Chem.Mol) -> Chem.Mol:
    """Apply the documented approximate pH 7.4 ionisation rule set.

    Edits are made in place on a copy and re-sanitised; if sanitisation fails
    the original molecule is returned untouched.
    """
    work = Chem.Mol(mol)
    changed = False

    for match in work.GetSubstructMatches(ACID_OH):
        changed |= _set_ion(work, match[0], -1, -1)
    for match in work.GetSubstructMatches(TETRAZOLE_NH):
        changed |= _set_ion(work, match[0], -1, -1)
        break

    basic_idx: set[int] = set()
    for match in work.GetSubstructMatches(AMIDINE_GUANIDINE):
        basic_idx.add(match[0])
    if not basic_idx:
        for match in work.GetSubstructMatches(ALIPHATIC_AMINE):
            atom = work.GetAtomWithIdx(match[0])
            if atom.GetIsAromatic() or atom.GetDegree() > 3:
                continue
            if any(nb.GetSymbol() != "C" or nb.GetIsAromatic() for nb in atom.GetNeighbors()):
                continue
            basic_idx.add(match[0])
    for idx in sorted(basic_idx)[:MAX_ABS_FORMAL_CHARGE]:
        changed |= _set_ion(work, idx, 1, 1)

    if not changed:
        return mol
    try:
        Chem.SanitizeMol(work)
    except Exception:  # noqa: BLE001
        return mol
    return work


def prepare_one(task: tuple) -> PrepResult:
    cand_id, smiles, seed, n_confs, max_heavy = task
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return PrepResult(cand_id, False, "SMILES_PARSE_FAILED")
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception as exc:  # noqa: BLE001
        return PrepResult(cand_id, False, f"STANDARDISE_FAILED:{type(exc).__name__}")
    if mol is None or mol.GetNumAtoms() == 0:
        return PrepResult(cand_id, False, "EMPTY_AFTER_STANDARDISE")
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 8:
        return PrepResult(cand_id, False, "TOO_FEW_HEAVY_ATOMS")
    if heavy > max_heavy:
        return PrepResult(cand_id, False, "TOO_MANY_HEAVY_ATOMS")
    if any(a.GetSymbol() not in {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Se"} for a in mol.GetAtoms()):
        return PrepResult(cand_id, False, "UNSUPPORTED_ELEMENT")

    try:
        mol = protonate_ph74(mol)
    except Exception:
        pass

    inchikey = ""
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        pass

    props = dict(
        mw=float(Descriptors.MolWt(mol)),
        clogp=float(Descriptors.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        hba=int(rdMolDescriptors.CalcNumHBA(mol)),
        hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
        rotb=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        fsp3=float(rdMolDescriptors.CalcFractionCSP3(mol)),
    )
    try:
        props["qed"] = float(QED.qed(mol))
    except Exception:
        props["qed"] = float("nan")
    try:
        murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        murcko = ""

    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(molh, numConfs=n_confs, params=params)
    if len(cids) == 0:
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(molh, numConfs=n_confs, params=params)
    if len(cids) == 0:
        return PrepResult(cand_id, False, "EMBED_FAILED")
    best_cid, best_energy = int(cids[0]), None
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(molh, maxIters=800, mmffVariant="MMFF94s")
        for cid, (converged, energy) in zip(cids, results):
            if best_energy is None or energy < best_energy:
                best_energy, best_cid = energy, int(cid)
    except Exception:
        pass
    conf_mol = Chem.Mol(molh)
    conf_mol.RemoveAllConformers()
    conf_mol.AddConformer(molh.GetConformer(best_cid), assignId=True)
    conf_mol.SetProp("_Name", cand_id)
    return PrepResult(
        cand_id=cand_id,
        ok=True,
        reason="OK",
        inchikey=inchikey,
        smiles_prepared=Chem.MolToSmiles(mol),
        formal_charge=int(Chem.GetFormalCharge(mol)),
        heavy_atoms=int(heavy),
        molblock=Chem.MolToMolBlock(conf_mol),
        murcko=murcko,
        **props,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(root: Path, denovo_queue: Path | None = None,
                sources: set[str] | None = None, id_prefix: str = "DN") -> pd.DataFrame:
    v2 = root / "YONSEI_FINAL10_V2"
    sources = sources or {"DENOVO", "SOURCEABLE"}
    queue_path = denovo_queue or (v2 / "02_DENOVO_CPU/DENOVO_TOP3000_FOR_DOCKING.csv")
    denovo = pd.read_csv(queue_path) if "DENOVO" in sources else pd.DataFrame()
    universe = (pd.read_csv(v2 / "03_SOURCEABILITY/V2_CANDIDATE_UNIVERSE.csv")
                if "SOURCEABLE" in sources else pd.DataFrame())

    rows = []
    for _, r in denovo.iterrows():
        target = str(r["dock_target"])
        if target == "BOTH_5NDD_AND_5NDZ":
            pockets = [POCKET_5NDD, POCKET_5NDZ]
        elif target == POCKET_5NDZ:
            pockets = [POCKET_5NDZ]
        else:
            pockets = [POCKET_5NDD]
        rows.append(
            dict(
                cand_id=f"{id_prefix}{int(r['dock_priority_rank']):05d}",
                source="DENOVO",
                smiles_input=r["smiles"],
                pockets=";".join(pockets),
                denovo_family=r["family"],
                denovo_rank=int(r["dock_priority_rank"]),
                design_score=float(r["design_score"]),
                ref_tanimoto=float(r["ref_tanimoto"]),
                library="DE_NOVO_BRICS",
                triage_class="DE_NOVO",
                candidate_name=f"denovo_{id_prefix}_{int(r['dock_priority_rank']):05d}",
                library_rank=int(getattr(r, "library_rank", r["dock_priority_rank"])),
                v1_rank="",
                sourceable_key="",
                tautomer_corrected=bool(getattr(r, "tautomer_corrected", False)),
            )
        )
    for i, r in universe.iterrows():
        rows.append(
            dict(
                cand_id=f"SU{i:05d}",
                source="SOURCEABLE",
                smiles_input=r["SMILES"],
                pockets=f"{POCKET_5NDD};{POCKET_5NDZ}",
                denovo_family="",
                denovo_rank="",
                design_score="",
                ref_tanimoto="",
                library=r.get("original_library", ""),
                triage_class=r.get("triage_class", ""),
                candidate_name=r.get("candidate_name", ""),
                v1_rank=r.get("original_v1_rank", ""),
                sourceable_key=r.get("candidate_key", ""),
                tautomer_corrected=False,
                library_rank="",
            )
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--n-confs", type=int, default=8)
    parser.add_argument("--max-heavy", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap ligand count")
    parser.add_argument("--denovo-queue", type=Path, default=None,
                        help="override the de novo docking queue CSV")
    parser.add_argument("--sources", default="DENOVO,SOURCEABLE",
                        help="which input sources to prepare")
    parser.add_argument("--id-prefix", default="DN",
                        help="cand_id prefix; use a distinct value per batch so ids never collide")
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / "YONSEI_FINAL10_V2/04_V2_SCREEN").resolve()
    lig_dir = out / "ligands"
    shard_dir = out / "shards"
    for d in (out, lig_dir, shard_dir):
        d.mkdir(parents=True, exist_ok=True)

    table = load_inputs(root, args.denovo_queue,
                        {x.strip().upper() for x in args.sources.split(",") if x.strip()},
                        args.id_prefix)
    if args.limit:
        table = pd.concat([table[table.source == "DENOVO"].head(args.limit),
                           table[table.source == "SOURCEABLE"].head(args.limit)])
    print(f"[prep] input rows: {len(table)}", flush=True)

    tasks = [(r.cand_id, r.smiles_input, args.seed, args.n_confs, args.max_heavy) for r in table.itertuples()]
    results: dict[str, PrepResult] = {}
    with mp.Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(prepare_one, tasks, chunksize=8), 1):
            results[res.cand_id] = res
            if n % 500 == 0:
                print(f"[prep] {n}/{len(tasks)}", flush=True)

    table["prep_ok"] = table.cand_id.map(lambda c: results[c].ok)
    table["prep_reason"] = table.cand_id.map(lambda c: results[c].reason)
    for field in ("inchikey", "smiles_prepared", "formal_charge", "heavy_atoms", "mw", "clogp",
                  "tpsa", "hba", "hbd", "rotb", "fsp3", "qed", "murcko"):
        table[field] = table.cand_id.map(lambda c, f=field: getattr(results[c], f))

    good = table[table.prep_ok].copy()
    # Deduplicate by prepared InChIKey; de novo yields to a sourceable twin.
    good["_pref"] = good.source.map({"SOURCEABLE": 0, "DENOVO": 1})
    good = good.sort_values(["inchikey", "_pref"]).drop_duplicates("inchikey", keep="first").drop(columns="_pref")
    dropped_dupes = int(table.prep_ok.sum() - len(good))
    print(f"[prep] prepared {len(good)} unique ligands ({dropped_dupes} duplicate InChIKeys collapsed)", flush=True)

    manifest_rows = []
    shard_index = {POCKET_5NDD: 0, POCKET_5NDZ: 0}
    buffers: dict[str, list[str]] = {POCKET_5NDD: [], POCKET_5NDZ: []}
    shard_files: dict[str, list[Path]] = {POCKET_5NDD: [], POCKET_5NDZ: []}

    def flush(pocket: str) -> None:
        if not buffers[pocket]:
            return
        path = shard_dir / f"{pocket}_shard{shard_index[pocket]:04d}.sdf"
        path.write_text("".join(buffers[pocket]))
        shard_files[pocket].append(path)
        buffers[pocket] = []
        shard_index[pocket] += 1

    for r in good.itertuples():
        res = results[r.cand_id]
        block = res.molblock.rstrip("\n")
        record = (
            f"{block}\n"
            f">  <cand_id>\n{r.cand_id}\n\n"
            f">  <source>\n{r.source}\n\n"
            f">  <inchikey>\n{res.inchikey}\n\n"
            f">  <smiles_prepared>\n{res.smiles_prepared}\n\n"
            f"$$$$\n"
        )
        for pocket in str(r.pockets).split(";"):
            buffers[pocket].append(record)
            manifest_rows.append(dict(cand_id=r.cand_id, pocket=pocket,
                                      shard=f"{pocket}_shard{shard_index[pocket]:04d}.sdf"))
            if len(buffers[pocket]) >= args.shard_size:
                flush(pocket)
    for pocket in (POCKET_5NDD, POCKET_5NDZ):
        flush(pocket)

    ligand_table = out / "V2_LIGAND_TABLE.csv"
    good.drop(columns=[]).to_csv(ligand_table, index=False)
    (out / "V2_PREP_FAILURES.csv").write_text(
        table[~table.prep_ok][["cand_id", "source", "smiles_input", "prep_reason"]].to_csv(index=False)
    )
    routing = pd.DataFrame(manifest_rows)
    routing.to_csv(out / "V2_DOCKING_ROUTING.csv", index=False)

    manifest = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        rdkit_version=Chem.rdBase.rdkitVersion,
        python=sys.version.split()[0],
        seed=args.seed,
        n_confs_embedded=args.n_confs,
        conformers_written_per_ligand=1,
        protonation_model="approximate pH 7.4 SMARTS rule set (documented in script; not a pKa prediction)",
        denovo_queue=str(args.denovo_queue) if args.denovo_queue else "default",
        sources=args.sources,
        id_prefix=args.id_prefix,
        tautomer_corrected_inputs=int(table.tautomer_corrected.sum())
        if "tautomer_corrected" in table else 0,
        input_rows=int(len(table)),
        prepared_ok=int(table.prep_ok.sum()),
        prepared_failed=int((~table.prep_ok).sum()),
        duplicate_inchikeys_collapsed=dropped_dupes,
        unique_ligands=int(len(good)),
        ligands_by_source={k: int(v) for k, v in good.source.value_counts().items()},
        docking_jobs_total=int(len(routing)),
        docking_jobs_by_pocket={k: int(v) for k, v in routing.pocket.value_counts().items()},
        shards={p: [f.name for f in shard_files[p]] for p in shard_files},
        shard_size=args.shard_size,
        shard_sha256={f.name: sha256(f) for p in shard_files for f in shard_files[p]},
        outputs={
            "V2_LIGAND_TABLE.csv": sha256(ligand_table),
            "V2_DOCKING_ROUTING.csv": sha256(out / "V2_DOCKING_ROUTING.csv"),
        },
        note="Ligand preparation only. No docking performed. No activity claim.",
    )
    (out / "V2_PREP_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"shards", "shard_sha256"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
