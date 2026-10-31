#!/usr/bin/env python3
"""
PAR2 / F2RL1 CPU DE NOVO DESIGN
================================

Goal
----
Generate ~15,000 chemically diverse SMALL-MOLECULE PAR2 designs on a 24-vCPU
Google Compute Engine VM, using the two experimentally validated antagonist
families already present in the project:

  5NDD / AZ8838  -> orthosteric / extracellular competitive-antagonist family
  5NDZ / AZ3451  -> extra-helical TM2-TM4 negative-allosteric family

Default design budget
---------------------
  10,000 orthosteric-like designs
   3,500 allosteric-like designs
   1,500 exploratory designs

Why asymmetric?
---------------
NotebookLM/literature synthesis gives the orthosteric family the stronger first-pass
case for a topical pigmentation program: AZ8838-like chemistry can competitively
block protease/tethered-ligand activation, while the 5NDZ family remains valuable as
an independent mechanism but is more vulnerable to lipophilicity/probe-dependence.
The exploratory bucket deliberately preserves uncertainty.

Important scientific limit
--------------------------
This script DOES NOT claim that generated molecules are antagonists. It only creates
chemistry that is compatible with the two antagonist design hypotheses. The real
residue-level test comes NEXT:

  corrected GNINA docking -> multiple poses -> pocket-specific ProLIF/PLIP IFP

Do not replace that with a 2D similarity score or a CNN 'probability'.

CPU strategy
------------
This script is intentionally CPU-native. It uses RDKit BRICS recombination because
15,000 PocketXMol diffusion samples on CPU would be poor ROI. The 24-core VM is used
for parallel medicinal-chemistry scoring. Default workers = 20, leaving headroom for
OS/filesystem work.

Inputs
------
The script auto-discovers:
  * clean crystallographic AZ8838 SDF under YONSEI_FINAL10_V2/01_BENCHMARK_REPAIR
  * clean crystallographic AZ3451 SDF under the same folder
  * V1_TOP500.csv, falling back to RESULTS/CNN_FINAL_SCORES.csv

V1 molecules are used only as FRAGMENT DONORS / diversity chemistry. Their old CNN
ranking is not inherited as biological truth.

Outputs
-------
  YONSEI_FINAL10_V2/02_DENOVO_CPU/
    DENOVO_15000.csv
    DENOVO_TOP3000_FOR_DOCKING.csv
    GENERATION_MANIFEST.json
    denovo.log

Smoke test
----------
  python par2_denovo_cpu.py --repo . --n-total 300 --top-docking 100 --workers 8

Full 24-core run
----------------
  python par2_denovo_cpu.py --repo . --n-total 15000 --top-docking 3000 --workers 20

Dependencies
------------
  python -m pip install "rdkit>=2025.03" pandas numpy tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold


# -----------------------------------------------------------------------------
# FAMILY-SPECIFIC DESIGN PRIORS
# -----------------------------------------------------------------------------
# These are SOFT medicinal-chemistry priors, not statements that one exact set of
# numbers is biologically required. Broad hard bounds preserve variance.
#
# Orthosteric / 5NDD logic from NotebookLM:
#   - favour non-planar / sp3-rich medicinal chemistry
#   - retain enough H-bond acceptors + aromatic/hydrophobic surface to reproduce
#     the ECL2/TM cleft pharmacophore after docking
#   - avoid peptide-like / overly flexible chemistry
#
# Allosteric / 5NDZ logic:
#   - membrane-facing pocket tolerates more lipophilicity
#   - but cap cLogP so we do not simply generate greasy Box-B scoring artefacts
#   - require enough heteroatom/polar character for directional recognition
# -----------------------------------------------------------------------------

FAMILY_CFG = {
    "orthosteric": {
        "target": dict(mw=360, logp=3.0, tpsa=65, hba=3.5, hbd=1.0,
                       rotb=4.0, fsp3=0.35, arom=2.0, refsim=0.38),
        "sigma":  dict(mw=85, logp=1.0, tpsa=25, hba=2.0, hbd=1.3,
                       rotb=3.0, fsp3=0.22, arom=1.2, refsim=0.20),
        "hard":   dict(mw=(230,500), logp=(1.0,5.0), tpsa=(25,115),
                       hba=(1,9), hbd=(0,3), rotb=(0,9), charge=(-1,1)),
    },
    "allosteric": {
        "target": dict(mw=375, logp=3.7, tpsa=52, hba=3.0, hbd=0.7,
                       rotb=3.5, fsp3=0.30, arom=2.0, refsim=0.35),
        "sigma":  dict(mw=90, logp=1.0, tpsa=24, hba=1.8, hbd=1.1,
                       rotb=2.8, fsp3=0.22, arom=1.2, refsim=0.20),
        "hard":   dict(mw=(240,510), logp=(1.8,5.5), tpsa=(18,100),
                       hba=(1,8), hbd=(0,3), rotb=(0,8), charge=(-1,1)),
    },
    "exploratory": {
        "target": dict(mw=350, logp=3.1, tpsa=62, hba=3.5, hbd=1.0,
                       rotb=4.0, fsp3=0.32, arom=2.0, refsim=0.25),
        "sigma":  dict(mw=110, logp=1.4, tpsa=34, hba=2.5, hbd=1.7,
                       rotb=4.0, fsp3=0.28, arom=1.5, refsim=0.25),
        "hard":   dict(mw=(200,520), logp=(0.5,5.7), tpsa=(15,125),
                       hba=(0,10), hbd=(0,4), rotb=(0,10), charge=(-1,1)),
    },
}

# Narrow set of clearly reactive groups. We intentionally do NOT apply an enormous
# blacklist that could throw away valid medicinal chemistry.
REACTIVE_SMARTS = {
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl,Br,I]",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "peroxide": "[OX2]-[OX2]",
    "diazo": "[N-]=[N+]=C",
}
REACTIVE = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE_SMARTS.items()}

# Multiprocessing globals, initialized once per worker.
MORGAN = None
REF_FP = {}
PAINS = None
GEN_POOL = None
GEN_SEED_FRAGS = {}
GEN_RXN_FRAGMENTS = None
GEN_SHARD_SECONDS = 20.0


def gaussian(x: float, target: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - target) / sigma) ** 2)


def init_worker(ref_smiles: Dict[str, str]) -> None:
    global MORGAN, REF_FP, PAINS
    MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    REF_FP = {}
    for family, smi in ref_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            REF_FP[family] = MORGAN.GetFingerprint(mol)

    params = FilterCatalogParams()
    for name in ("PAINS_A", "PAINS_B", "PAINS_C"):
        enum = getattr(FilterCatalogParams.FilterCatalogs, name, None)
        if enum is not None:
            params.AddCatalog(enum)
    PAINS = FilterCatalog(params)


def init_generation_worker(fragment_pool: Sequence[Chem.Mol],
                           family_seed_frags: Dict[str, Sequence[Chem.Mol]],
                           shard_seconds: float) -> None:
    """Install read-only BRICS inputs once in each forked generation worker."""
    global GEN_POOL, GEN_SEED_FRAGS, GEN_RXN_FRAGMENTS, GEN_SHARD_SECONDS
    GEN_POOL = list(fragment_pool)
    GEN_SEED_FRAGS = {family: list(frags) for family, frags in family_seed_frags.items()}
    GEN_SHARD_SECONDS = shard_seconds
    # RDKit's stock BRICSBuild rescans every fragment against both reaction
    # matchers at every recursion. Cache those invariant matches once.
    GEN_RXN_FRAGMENTS = []
    for reaction in BRICS.reverseReactions:
        compatible = []
        for fragment in GEN_POOL:
            matches_left = fragment.HasSubstructMatch(reaction._matchers[0])
            matches_right = fragment.HasSubstructMatch(reaction._matchers[1])
            if matches_left or matches_right:
                compatible.append((fragment, matches_left, matches_right))
        GEN_RXN_FRAGMENTS.append(compatible)


def fast_brics_build(fragments: Sequence[Chem.Mol], only_complete_mols=True,
                     seeds=None, uniquify=True, scramble_reagents=True,
                     max_depth=3, deadline=None):
    """Chemistry-equivalent BRICSBuild with cached fragment/matcher lookups."""
    seen = set()
    if not seeds:
        seeds = list(fragments)
    else:
        seeds = list(seeds)
    if scramble_reagents:
        random.shuffle(seeds)
        reaction_indices = list(range(len(BRICS.reverseReactions)))
        random.shuffle(reaction_indices)
    else:
        reaction_indices = list(range(len(BRICS.reverseReactions)))

    for seed_mol in seeds:
        if deadline is not None and time.monotonic() >= deadline:
            return
        seed_is_left = False
        seed_is_right = False
        next_steps = []
        for reaction_index in reaction_indices:
            if deadline is not None and time.monotonic() >= deadline:
                return
            reaction = BRICS.reverseReactions[reaction_index]
            if seed_mol.HasSubstructMatch(reaction._matchers[0]):
                seed_is_left = True
            if seed_mol.HasSubstructMatch(reaction._matchers[1]):
                seed_is_right = True
            for fragment, matches_left, matches_right in GEN_RXN_FRAGMENTS[reaction_index]:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                product_sets = None
                if matches_left and seed_is_right:
                    product_sets = reaction.RunReactants((fragment, seed_mol))
                # Preserve RDKit BRICSBuild's ordering/overwrite behavior when
                # a fragment happens to match both sides of a reaction.
                if matches_right and seed_is_left:
                    product_sets = reaction.RunReactants((seed_mol, fragment))
                if not product_sets:
                    continue
                for product_tuple in product_sets:
                    product = product_tuple[0]
                    product_smiles = Chem.MolToSmiles(product, True) if uniquify else None
                    if uniquify and product_smiles in seen:
                        continue
                    if uniquify:
                        seen.add(product_smiles)
                    if product.HasSubstructMatch(BRICS.dummyPattern):
                        next_steps.append(product)
                        if not only_complete_mols:
                            yield product
                    else:
                        yield product
        if next_steps and max_depth > 0:
            for product in fast_brics_build(
                fragments,
                only_complete_mols=only_complete_mols,
                seeds=next_steps,
                uniquify=uniquify,
                scramble_reagents=scramble_reagents,
                max_depth=max_depth - 1,
                deadline=deadline,
            ):
                product_smiles = Chem.MolToSmiles(product, True) if uniquify else None
                if uniquify and product_smiles in seen:
                    continue
                if uniquify:
                    seen.add(product_smiles)
                yield product


def discover_csv(repo: Path) -> Path:
    for p in [
        repo / "YONSEI_FINAL10_V2/00_AUDIT/V1_TOP500.csv",
        repo / "RESULTS/CNN_FINAL_SCORES.csv",
    ]:
        if p.exists() and p.stat().st_size:
            return p
    raise FileNotFoundError("Cannot find V1_TOP500.csv or RESULTS/CNN_FINAL_SCORES.csv")


def discover_seed(repo: Path, family: str) -> Path:
    # Prefer the cleaned crystallographic inputs.  Benchmark-repair contains
    # fresh redocking outputs, which are validation results rather than seeds.
    clean_root = repo / "YONSEI_FINAL10_V2/01_REFERENCE_POCKETS/ligands"
    expected = (
        clean_root / "5NDD_AZ8838_crystallographic_clean.sdf"
        if family == "orthosteric"
        else clean_root / "5NDZ_AZ3451_crystallographic_clean.sdf"
    )
    if expected.exists() and expected.stat().st_size:
        return expected

    roots = [
        clean_root,
        repo / "YONSEI_FINAL10_V2/01_BENCHMARK_REPAIR",
    ]
    roots = [root for root in roots if root.exists()]
    if not roots:
        raise FileNotFoundError("Missing V2 reference ligand directories")

    tokens = ("az8838", "8tz", "5ndd") if family == "orthosteric" else ("az3451", "8un", "5ndz")
    candidates = []
    for root in roots:
        for p in root.rglob("*.sdf"):
            txt = str(p).lower()
            if not any(t in txt for t in tokens):
                continue
            score = 0
            if any(t in txt for t in ("clean", "reference", "crystal", "native")):
                score += 5
            if any(t in txt for t in ("redock", "poses", "output", "out_")):
                score -= 8
            candidates.append((score, p))

    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-find clean {family} SDF. Pass --{family}-seed-sdf explicitly."
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_sdf_first(path: Path) -> Chem.Mol:
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if not mols:
        raise ValueError(f"No valid molecule in {path}")
    mol = mols[0]
    Chem.SanitizeMol(mol)
    return mol


def load_donors(path: Path, n: int) -> List[Chem.Mol]:
    df = pd.read_csv(path)
    col = next((c for c in ("smiles", "SMILES", "canonical_smiles") if c in df.columns), None)
    if not col:
        raise ValueError(f"No SMILES column in {path}; columns={list(df.columns)}")
    out = []
    for smi in df[col].dropna().astype(str).head(n):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            out.append(mol)
    return out


def brics_fragments(mols: Sequence[Chem.Mol], min_heavy=3, max_heavy=16) -> List[Chem.Mol]:
    unique = {}
    for mol in mols:
        try:
            frags = BRICS.BRICSDecompose(mol, returnMols=True, minFragmentSize=min_heavy)
        except Exception:
            continue
        for frag in frags:
            heavy = sum(a.GetAtomicNum() > 1 for a in frag.GetAtoms())
            if min_heavy <= heavy <= max_heavy:
                unique[Chem.MolToSmiles(frag, True)] = frag
    return list(unique.values())


def canonical(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
        if not (12 <= mol.GetNumHeavyAtoms() <= 44):
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def generate_raw(family: str, n_raw: int, pool: Sequence[Chem.Mol],
                 seed_frags: Sequence[Chem.Mol], seed: int) -> List[str]:
    random.seed(seed)
    result = {}
    deadline = time.monotonic() + GEN_SHARD_SECONDS

    # Family designs are anchored to at least one fragment from the crystal ligand.
    # Exploration deliberately removes that requirement.
    seeds = None if family == "exploratory" else list(seed_frags)

    for depth in (3, 2, 4):
        if len(result) >= n_raw:
            break
        try:
            gen = fast_brics_build(
                list(pool), seeds=seeds, max_depth=depth,
                only_complete_mols=True, uniquify=True, scramble_reagents=True,
                deadline=deadline,
            )
            for mol in gen:
                smi = canonical(mol)
                if smi:
                    result[smi] = None
                if len(result) >= n_raw:
                    break
        except Exception as exc:
            logging.warning("BRICS %s depth=%d failed: %r", family, depth, exc)

    # If reference-anchored generation cannot fill the quota, allow unseeded
    # recombination rather than silently returning an undersized campaign.
    if len(result) < n_raw and family != "exploratory" and time.monotonic() < deadline:
        logging.warning("%s seeded generation only made %d/%d; filling unseeded", family, len(result), n_raw)
        try:
            gen = fast_brics_build(list(pool), max_depth=3, only_complete_mols=True,
                                   uniquify=True, scramble_reagents=True,
                                   deadline=deadline)
            for mol in gen:
                smi = canonical(mol)
                if smi:
                    result[smi] = None
                if len(result) >= n_raw:
                    break
        except Exception as exc:
            logging.warning("Unseeded fallback failed: %r", exc)

    return list(result)


def generate_raw_shard(task: Tuple[str, int, int]) -> Tuple[str, List[str]]:
    """Generate one independently seeded shard using the unchanged BRICS rules."""
    family, n_raw, seed = task
    seed_frags = GEN_SEED_FRAGS.get(family, [])
    return family, generate_raw(family, n_raw, GEN_POOL, seed_frags, seed)


def generate_all_families_parallel(targets: Dict[str, int], oversample: float,
                                   fragment_pool: Sequence[Chem.Mol],
                                   family_seed_frags: Dict[str, Sequence[Chem.Mol]],
                                   seed: int, workers: int,
                                   shard_seconds: float) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Parallelize BRICS enumeration without changing chemistry or scoring.

    Shards differ only in RNG seed. Their canonical SMILES are merged and
    deduplicated before the existing scoring/filtering stage.
    """
    desired = {
        family: max(target, math.ceil(target * oversample))
        for family, target in targets.items()
    }
    # BRICS enumeration has an expensive long tail when one generator is asked
    # for thousands of unique products. Keep shards near 500 products and let
    # the 20-core pool process many deterministic shards instead.
    shard_counts = {
        family: min(200, max(2, math.ceil(n_raw / 500)))
        for family, n_raw in desired.items()
    }
    generated = {family: {} for family in desired}
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    generation_workers = min(workers, sum(shard_counts.values()))
    logging.info("Parallel BRICS generation workers=%d shards=%s", generation_workers, shard_counts)

    with ctx.Pool(
        generation_workers,
        initializer=init_generation_worker,
        initargs=(fragment_pool, family_seed_frags, shard_seconds),
    ) as gen_pool:
        for round_index in range(4):
            tasks = []
            for family, n_raw in desired.items():
                missing = n_raw - len(generated[family])
                if missing <= 0:
                    continue
                shards = shard_counts[family]
                # Modest duplicate allowance; later rounds refill only deficits.
                per_shard = max(1, math.ceil(missing * 1.20 / shards))
                family_offset = ("orthosteric", "allosteric", "exploratory").index(family) * 100000
                for shard_index in range(shards):
                    shard_seed = seed + family_offset + round_index * 10000 + shard_index * 101
                    tasks.append((family, per_shard, shard_seed))
            if not tasks:
                break
            logging.info("BRICS round %d: %d shard tasks", round_index + 1, len(tasks))
            for family, smiles in gen_pool.imap(generate_raw_shard, tasks, chunksize=1):
                generated[family].update(dict.fromkeys(smiles))
                if len(generated[family]) > desired[family]:
                    # Dict insertion order makes this deterministic for each
                    # completed task stream while bounding memory.
                    generated[family] = dict(list(generated[family].items())[:desired[family]])
            logging.info("BRICS round %d totals: %s", round_index + 1,
                         {family: len(values) for family, values in generated.items()})

    raw_tasks = []
    raw_counts = {}
    for family in ("orthosteric", "allosteric", "exploratory"):
        smiles = list(generated[family])[:desired[family]]
        raw_counts[family] = len(smiles)
        if len(smiles) < desired[family]:
            logging.warning("%s raw generation supplied %d/%d", family, len(smiles), desired[family])
        raw_tasks.extend((smi, family) for smi in smiles)
    return raw_tasks, raw_counts


def descriptors(m: Chem.Mol) -> Dict[str, float]:
    return {
        "mw": float(Descriptors.MolWt(m)),
        "logp": float(Crippen.MolLogP(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "hba": int(Lipinski.NumHAcceptors(m)),
        "hbd": int(Lipinski.NumHDonors(m)),
        "rotb": int(Lipinski.NumRotatableBonds(m)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(m)),
        "rings": int(rdMolDescriptors.CalcNumRings(m)),
        "arom": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "charge": int(Chem.GetFormalCharge(m)),
        "qed": float(QED.qed(m)),
    }


def hard_pass(d: Dict[str, float], family: str) -> bool:
    h = FAMILY_CFG[family]["hard"]
    for key in ("mw", "logp", "tpsa", "hba", "hbd", "rotb", "charge"):
        lo, hi = h[key]
        if not lo <= d[key] <= hi:
            return False
    return d["rings"] <= 6


def score_one(task: Tuple[str, str]) -> Optional[Dict]:
    smi, family = task
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    # Clear reactive groups.
    for patt in REACTIVE.values():
        if patt is not None and mol.HasSubstructMatch(patt):
            return None

    # Hard PAINS reject for this first wet-lab-oriented campaign.
    if PAINS is not None and PAINS.HasMatch(mol):
        return None

    d = descriptors(mol)
    if not hard_pass(d, family):
        return None

    fp = MORGAN.GetFingerprint(mol)
    if family == "orthosteric":
        refsim = DataStructs.TanimotoSimilarity(fp, REF_FP["orthosteric"])
    elif family == "allosteric":
        refsim = DataStructs.TanimotoSimilarity(fp, REF_FP["allosteric"])
    else:
        refsim = max(
            DataStructs.TanimotoSimilarity(fp, REF_FP["orthosteric"]),
            DataStructs.TanimotoSimilarity(fp, REF_FP["allosteric"]),
        )

    cfg = FAMILY_CFG[family]
    terms = []
    mapping = {"mw":"mw", "logp":"logp", "tpsa":"tpsa", "hba":"hba",
               "hbd":"hbd", "rotb":"rotb", "fsp3":"fsp3", "arom":"arom"}
    for k in mapping:
        terms.append(gaussian(d[mapping[k]], cfg["target"][k], cfg["sigma"][k]))
    property_score = float(np.mean(terms))

    # Goldilocks novelty: not a clone, not chemically unrelated.
    novelty = gaussian(refsim, cfg["target"]["refsim"], cfg["sigma"]["refsim"])

    # Count-level feature prior only. Residue-specific features are intentionally
    # deferred to docking/IFP so we do not pretend 2D descriptors know 3D contacts.
    if family == "orthosteric":
        feature = float(np.mean([
            gaussian(d["hba"], 3.5, 2.0),
            gaussian(d["arom"], 2.0, 1.2),
            gaussian(d["fsp3"], 0.35, 0.22),
        ]))
    elif family == "allosteric":
        feature = float(np.mean([
            gaussian(d["hba"], 3.0, 1.8),
            gaussian(d["arom"], 2.0, 1.2),
            gaussian(d["logp"], 3.7, 1.0),
            gaussian(d["fsp3"], 0.30, 0.22),
        ]))
    else:
        feature = float(np.mean([
            gaussian(d["hba"], 3.5, 2.5),
            gaussian(d["fsp3"], 0.32, 0.28),
        ]))

    score = 0.50 * property_score + 0.20 * d["qed"] + 0.20 * novelty + 0.10 * feature
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True) or smi

    return {
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True),
        "family": family,
        "design_score": score,
        "ref_tanimoto": refsim,
        "scaffold": scaffold,
        **d,
    }


def scaffold_select(rows: List[Dict], n: int, cap: int) -> List[Dict]:
    """Keep high-scoring chemistry while preventing one scaffold from taking over."""
    rows = sorted(rows, key=lambda r: r["design_score"], reverse=True)
    counts = Counter()
    keep, overflow = [], []
    for row in rows:
        sc = row["scaffold"]
        if counts[sc] < cap:
            keep.append(row)
            counts[sc] += 1
        else:
            overflow.append(row)
        if len(keep) == n:
            return keep
    for row in overflow:
        keep.append(row)
        if len(keep) == n:
            break
    return keep


def score_tasks_parallel(raw_tasks: List[Tuple[str, str]], workers: int,
                         ref_smiles: Dict[str, str]) -> List[Dict]:
    if not raw_tasks:
        return []
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    logging.info("Scoring %d candidates", len(raw_tasks))
    with ctx.Pool(workers, initializer=init_worker, initargs=(ref_smiles,)) as poolproc:
        scored = list(tqdm(poolproc.imap_unordered(score_one, raw_tasks, chunksize=128),
                           total=len(raw_tasks), desc="PAR2 de novo scoring"))
    return [row for row in scored if row is not None]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--donor-csv", type=Path, default=None)
    p.add_argument("--orthosteric-seed-sdf", type=Path, default=None)
    p.add_argument("--allosteric-seed-sdf", type=Path, default=None)
    p.add_argument("--n-total", type=int, default=15000)
    p.add_argument("--top-docking", type=int, default=3000)
    p.add_argument("--oversample", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--donor-rows", type=int, default=500)
    p.add_argument("--max-per-scaffold", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--generation-shard-seconds", type=float, default=20.0)
    p.add_argument("--max-filter-refill-rounds", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    outdir = (args.outdir.resolve() if args.outdir else repo / "YONSEI_FINAL10_V2/02_DENOVO_CPU")
    outdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(outdir / "denovo.log")],
    )

    cpus = os.cpu_count() or 1
    workers = args.workers or max(1, min(20, cpus - 2))
    logging.info("Detected %d CPUs; using %d workers", cpus, workers)

    # Preserve 10k / 3.5k / 1.5k proportions even for smoke tests.
    n_orth = round(args.n_total * 10000 / 15000)
    n_allo = round(args.n_total * 3500 / 15000)
    n_exp = args.n_total - n_orth - n_allo
    targets = {"orthosteric": n_orth, "allosteric": n_allo, "exploratory": n_exp}
    logging.info("Targets: %s", targets)

    donor_csv = args.donor_csv.resolve() if args.donor_csv else discover_csv(repo)
    orth_sdf = args.orthosteric_seed_sdf.resolve() if args.orthosteric_seed_sdf else discover_seed(repo, "orthosteric")
    allo_sdf = args.allosteric_seed_sdf.resolve() if args.allosteric_seed_sdf else discover_seed(repo, "allosteric")

    logging.info("Donor CSV: %s", donor_csv)
    logging.info("AZ8838 seed: %s", orth_sdf)
    logging.info("AZ3451 seed: %s", allo_sdf)

    orth = load_sdf_first(orth_sdf)
    allo = load_sdf_first(allo_sdf)
    donors = load_donors(donor_csv, args.donor_rows)

    orth_smiles = Chem.MolToSmiles(Chem.RemoveHs(orth), True)
    allo_smiles = Chem.MolToSmiles(Chem.RemoveHs(allo), True)

    pool = brics_fragments([orth, allo] + donors)
    orth_frags = brics_fragments([orth], min_heavy=2, max_heavy=20)
    allo_frags = brics_fragments([allo], min_heavy=2, max_heavy=20)
    if len(pool) < 20:
        raise RuntimeError(f"Only {len(pool)} BRICS fragments; inspect seed/donor chemistry")

    logging.info("Fragment pool=%d; orth seed fragments=%d; allo seed fragments=%d", len(pool), len(orth_frags), len(allo_frags))

    raw_tasks, raw_counts = generate_all_families_parallel(
        targets,
        args.oversample,
        pool,
        {"orthosteric": orth_frags, "allosteric": allo_frags, "exploratory": []},
        args.seed,
        workers,
        args.generation_shard_seconds,
    )

    ref_smiles = {"orthosteric": orth_smiles, "allosteric": allo_smiles}
    scored = score_tasks_parallel(raw_tasks, workers, ref_smiles)
    seen_raw_tasks = set(raw_tasks)
    aggregate_raw_counts = dict(raw_counts)
    refill_history = []

    # Keep every chemistry/filter rule unchanged while filling families that
    # remain short after hard filtering. New deterministic seeds create only
    # the needed family chemistry instead of rerunning the whole campaign.
    for refill_round in range(args.max_filter_refill_rounds):
        available = defaultdict(set)
        for row in scored:
            available[row["family"]].add(row["smiles"])
        deficits = {
            family: max(0, target - len(available[family]))
            for family, target in targets.items()
        }
        if not any(deficits.values()):
            break
        logging.info("Filtered quota refill %d deficits=%s", refill_round + 1, deficits)
        refill_tasks, refill_raw_counts = generate_all_families_parallel(
            deficits,
            max(10.0, args.oversample),
            pool,
            {"orthosteric": orth_frags, "allosteric": allo_frags, "exploratory": []},
            args.seed + (refill_round + 1) * 1000000,
            workers,
            args.generation_shard_seconds,
        )
        refill_tasks = [task for task in refill_tasks if task not in seen_raw_tasks]
        seen_raw_tasks.update(refill_tasks)
        refill_scored = score_tasks_parallel(refill_tasks, workers, ref_smiles)
        scored.extend(refill_scored)
        for family, count in refill_raw_counts.items():
            aggregate_raw_counts[family] = aggregate_raw_counts.get(family, 0) + count
        refill_history.append({
            "round": refill_round + 1,
            "deficits_before": deficits,
            "new_raw_unique": len(refill_tasks),
            "new_filtered": len(refill_scored),
        })

    # Preserve the expensive filtered pool for audit and recovery.
    pd.DataFrame(scored).to_csv(outdir / "SCORED_CANDIDATES.csv", index=False)

    by_family = defaultdict(list)
    for row in scored:
        by_family[row["family"]].append(row)

    # Assign scarce families first and enforce exact-SMILES uniqueness while
    # selecting, so cross-family duplicates are backfilled rather than silently
    # shrinking the requested 15,000-molecule output.
    final = []
    selected_smiles = set()
    selected_counts = {}
    family_order = sorted(targets, key=lambda family: len(by_family[family]) - targets[family])
    for family in family_order:
        n = targets[family]
        ranked = scaffold_select(by_family[family], len(by_family[family]), args.max_per_scaffold)
        chosen = []
        for row in ranked:
            if row["smiles"] in selected_smiles:
                continue
            chosen.append(row)
            selected_smiles.add(row["smiles"])
            if len(chosen) == n:
                break
        selected_counts[family] = len(chosen)
        if len(chosen) < n:
            logging.warning("%s only supplied %d/%d unique filtered designs", family, len(chosen), n)
        final.extend(chosen)
    final = sorted(final, key=lambda r: r["design_score"], reverse=True)

    if len(final) != args.n_total:
        raise RuntimeError(
            f"Generated {len(final)}/{args.n_total} unique filtered molecules; "
            "rerun with a larger --oversample value"
        )

    df = pd.DataFrame(final)
    df.insert(0, "denovo_rank", range(1, len(df)+1))
    df.to_csv(outdir / "DENOVO_15000.csv", index=False)

    # The docking set is still scaffold-capped. We are NOT simply sending the top
    # 3,000 numerical scores, because that would collapse chemical diversity.
    top = scaffold_select(final, min(args.top_docking, len(final)), max(5, args.max_per_scaffold // 2))
    top = sorted(top, key=lambda r: r["design_score"], reverse=True)
    dft = pd.DataFrame(top)
    dft.insert(0, "dock_priority_rank", range(1, len(dft)+1))
    dft["dock_target"] = dft["family"].map({
        "orthosteric": "5NDD_orthosteric",
        "allosteric": "5NDZ_allosteric",
        "exploratory": "BOTH_5NDD_AND_5NDZ",
    })
    dft.to_csv(outdir / "DENOVO_TOP3000_FOR_DOCKING.csv", index=False)

    manifest = {
        "requested_total": args.n_total,
        "actual_total_after_cross_family_dedup": len(final),
        "family_targets": targets,
        "raw_generated": aggregate_raw_counts,
        "filter_refill_history": refill_history,
        "filtered_by_family": {family: len(rows) for family, rows in by_family.items()},
        "selected_by_family": selected_counts,
        "top_for_docking": len(top),
        "workers": workers,
        "detected_cpus": cpus,
        "seed": args.seed,
        "oversample": args.oversample,
        "generation_shard_seconds": args.generation_shard_seconds,
        "max_filter_refill_rounds": args.max_filter_refill_rounds,
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution_optimizations": "Linux fork scoring pool; parallel deterministic BRICS shards; cached BRICS fragment/reaction matcher index; bounded refillable shard runtime.",
        "donor_csv": str(donor_csv),
        "orthosteric_seed_sdf": str(orth_sdf),
        "allosteric_seed_sdf": str(allo_sdf),
        "fragment_pool_size": len(pool),
        "scientific_guardrail": "No generated molecule is called an antagonist until corrected docking + pocket-specific IFP + wet lab.",
        "next_step": "Dock orthosteric->5NDD, allosteric->5NDZ, exploratory->both; retain multiple poses; rank with GNINA + IFP; similarity-search purchasable analogues only for strongest inaccessible winners.",
    }
    (outdir / "GENERATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    logging.info("DONE: %d final designs; %d queued for docking", len(final), len(top))
    logging.info("%s", outdir / "DENOVO_15000.csv")
    logging.info("%s", outdir / "DENOVO_TOP3000_FOR_DOCKING.csv")
    logging.info("%s", outdir / "GENERATION_MANIFEST.json")


if __name__ == "__main__":
    main()
