#!/usr/bin/env python3
"""Map de novo seed chemistry onto real, existing compounds.

For each similarity-search seed, queries public compound databases that expose
a programmatic similarity endpoint (PubChem PUG-REST fastsimilarity_2d and the
ChEMBL similarity API), then re-scores every returned structure locally with
RDKit Morgan/ECFP4 radius-2 2048-bit Tanimoto so the reported similarity is
reproducible and independent of each database's own fingerprint.

Bands: >=0.80 very close analogue, 0.70-0.80 close analogue,
0.55-0.70 broader scaffold/feature neighbour.

A lower-Tanimoto hit is kept, not discarded, when its scaffold relationship,
MCS overlap or 3D shape similarity is strong; those columns are computed so the
next docking gate can make that call on evidence.

This retrieves candidate structures only. It confirms nothing about PAR2
activity and does not establish that any hit is purchasable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "PAR2-V2-analogue-search/1.0 (research use)"

BANDS = [(0.80, 1.01, "very_close_analogue"),
         (0.70, 0.80, "close_analogue"),
         (0.55, 0.70, "broader_neighbour")]


def band_for(tanimoto: float) -> str:
    for low, high, name in BANDS:
        if low <= tanimoto < high:
            return name
    return "below_band"


def http_get(url: str, retries: int = 4, pause: float = 0.34) -> str | None:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                time.sleep(pause)
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (429, 503) and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"[warn] HTTP {exc.code} for {url[:110]}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"[warn] {type(exc).__name__} for {url[:110]}", file=sys.stderr)
            return None
    return None


def pubchem_similar(smiles: str, threshold_pct: int, max_records: int) -> list[dict]:
    quoted = urllib.parse.quote(smiles, safe="")
    url = (f"{PUBCHEM}/compound/fastsimilarity_2d/smiles/{quoted}/property/"
           f"CanonicalSMILES,IsomericSMILES,MolecularFormula,Title/JSON"
           f"?Threshold={threshold_pct}&MaxRecords={max_records}")
    body = http_get(url)
    if not body:
        return []
    try:
        properties = json.loads(body)["PropertyTable"]["Properties"]
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for record in properties:
        structure = next((record[k] for k in ("IsomericSMILES", "SMILES",
                                              "ConnectivitySMILES", "CanonicalSMILES")
                          if record.get(k)), None)
        if not structure:
            continue
        rows.append(dict(source_database="PubChem", source_compound_id=f"CID{record['CID']}",
                         source_name=record.get("Title", ""), hit_smiles=structure))
    return rows


def chembl_similar(smiles: str, threshold_pct: int, max_records: int) -> list[dict]:
    quoted = urllib.parse.quote(smiles, safe="")
    url = (f"{CHEMBL}/similarity/{quoted}/{threshold_pct}.json"
           f"?limit={min(max_records, 100)}"
           f"&only=molecule_chembl_id,pref_name,molecule_structures,max_phase")
    body = http_get(url)
    if not body:
        return []
    try:
        molecules = json.loads(body).get("molecules", [])
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for record in molecules:
        structures = record.get("molecule_structures") or {}
        structure = structures.get("canonical_smiles")
        if not structure:
            continue
        rows.append(dict(source_database="ChEMBL",
                         source_compound_id=record.get("molecule_chembl_id", ""),
                         source_name=record.get("pref_name") or "",
                         hit_smiles=structure,
                         chembl_max_phase=record.get("max_phase")))
    return rows


def embed(mol: Chem.Mol, seed: int = 0xF00D) -> Chem.Mol | None:
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(molh, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(molh, maxIters=400)
    except Exception:  # noqa: BLE001
        pass
    return molh


def mcs_similarity(seed_mol: Chem.Mol, hit_mol: Chem.Mol, timeout: int = 3) -> float | None:
    try:
        result = rdFMCS.FindMCS([seed_mol, hit_mol], timeout=timeout,
                                ringMatchesRingOnly=True, completeRingsOnly=True)
        if result.canceled or result.numAtoms == 0:
            return None
        union = seed_mol.GetNumHeavyAtoms() + hit_mol.GetNumHeavyAtoms() - result.numAtoms
        return result.numAtoms / union if union else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-seed-per-source", type=int, default=60)
    parser.add_argument("--min-tanimoto", type=float, default=0.55)
    parser.add_argument("--query-threshold-pct", type=int, default=55)
    parser.add_argument("--sources", default="pubchem,chembl")
    parser.add_argument("--shape-similarity", action="store_true",
                        help="also compute USRCAT 3D shape similarity (slower)")
    parser.add_argument("--no-network", action="store_true",
                        help="write the query set only; run no database queries")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_csv(args.seeds)
    if "canonical_smiles" not in seeds.columns:
        raise SystemExit("seeds file needs a canonical_smiles column")

    # Always emit the plain query set so the same seeds can be searched by hand
    # in any vendor catalogue that has no programmatic interface.
    seeds[[c for c in ["seed_id", "family", "pocket", "canonical_smiles", "postdock_rank",
                       "cluster_id", "murcko_scaffold"] if c in seeds.columns]].to_csv(
        out / "SEED_QUERY_SET.csv", index=False)
    if args.no_network:
        print(f"wrote {out / 'SEED_QUERY_SET.csv'}; no queries run (--no-network)")
        return 0

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    rows: list[dict] = []
    per_seed: dict[str, dict] = {}

    for seed in seeds.itertuples():
        seed_smiles = seed.canonical_smiles
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if seed_mol is None:
            print(f"[warn] unparsable seed {seed.seed_id}", file=sys.stderr)
            continue
        seed_fp = MORGAN.GetFingerprint(seed_mol)
        seed_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=seed_mol, includeChirality=False)
        seed_3d = embed(seed_mol) if args.shape_similarity else None
        seed_usrcat = (rdMolDescriptors.GetUSRCAT(seed_3d)
                       if (seed_3d is not None and seed_3d.GetNumConformers()) else None)

        raw: list[dict] = []
        if "pubchem" in sources:
            raw += pubchem_similar(seed_smiles, args.query_threshold_pct, args.max_per_seed_per_source)
        if "chembl" in sources:
            raw += chembl_similar(seed_smiles, args.query_threshold_pct, args.max_per_seed_per_source)

        kept = 0
        for hit in raw:
            mol = Chem.MolFromSmiles(hit["hit_smiles"])
            if mol is None:
                continue
            tanimoto = DataStructs.TanimotoSimilarity(seed_fp, MORGAN.GetFingerprint(mol))
            if tanimoto < args.min_tanimoto:
                continue
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:  # noqa: BLE001
                scaffold = ""
            shape = None
            if args.shape_similarity and seed_usrcat is not None:
                hit_3d = embed(mol)
                if hit_3d is not None and hit_3d.GetNumConformers():
                    try:
                        shape = rdMolDescriptors.GetUSRScore(
                            seed_usrcat, rdMolDescriptors.GetUSRCAT(hit_3d))
                    except Exception:  # noqa: BLE001
                        shape = None
            rows.append(dict(
                seed_id=seed.seed_id,
                seed_cand_id=getattr(seed, "cand_id", ""),
                seed_pocket=getattr(seed, "pocket", ""),
                seed_family=getattr(seed, "family", ""),
                seed_smiles=seed_smiles,
                source_database=hit["source_database"],
                source_compound_id=hit["source_compound_id"],
                source_name=hit.get("source_name", ""),
                chembl_max_phase=hit.get("chembl_max_phase"),
                hit_smiles=Chem.MolToSmiles(mol),
                inchikey=Chem.MolToInchiKey(mol),
                tanimoto_to_seed=round(tanimoto, 4),
                similarity_band=band_for(tanimoto),
                murcko_scaffold=scaffold,
                same_scaffold_as_seed=bool(scaffold and scaffold == seed_scaffold),
                mcs_similarity=mcs_similarity(seed_mol, mol),
                usrcat_shape_similarity=shape,
                vendor="PENDING", catalogue_ID="PENDING", approximate_price="PENDING",
                stock_status="PENDING",
            ))
            kept += 1
        per_seed[seed.seed_id] = dict(returned=len(raw), kept=kept)
        print(f"[sim] {seed.seed_id}: {len(raw)} returned, {kept} kept at "
              f">= {args.min_tanimoto}", flush=True)

    raw_frame = pd.DataFrame(rows)
    raw_frame.to_csv(out / "SIMILARITY_HITS_RAW.csv", index=False)

    if raw_frame.empty:
        dedup = raw_frame
    else:
        dedup = (raw_frame.sort_values("tanimoto_to_seed", ascending=False)
                 .drop_duplicates("inchikey", keep="first").reset_index(drop=True))
        counts = raw_frame.groupby("inchikey").seed_id.nunique()
        dedup["n_seeds_matching"] = dedup.inchikey.map(counts)
    dedup.to_csv(out / "SIMILARITY_HITS_DEDUP.csv", index=False)

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        seeds_queried=int(len(per_seed)),
        sources=sorted(sources),
        query_threshold_pct=args.query_threshold_pct,
        min_local_tanimoto=args.min_tanimoto,
        hits_raw=int(len(raw_frame)),
        hits_unique=int(len(dedup)),
        hits_per_seed=per_seed,
        band_counts={k: int(v) for k, v in raw_frame.similarity_band.value_counts().items()}
        if not raw_frame.empty else {},
        source_counts={k: int(v) for k, v in raw_frame.source_database.value_counts().items()}
        if not raw_frame.empty else {},
        caveat=("Structures retrieved from public databases. Vendor, catalogue ID, price and "
                "stock are PENDING and were not looked up. No PAR2 activity is implied."),
    )
    (out / "SIMILARITY_SEARCH_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "hits_per_seed"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
