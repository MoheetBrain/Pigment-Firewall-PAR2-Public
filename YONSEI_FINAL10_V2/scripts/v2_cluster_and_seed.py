#!/usr/bin/env python3
"""Cluster the best post-docking de novo designs and choose similarity-search seeds.

Butina clustering on Morgan/ECFP4 fingerprints (radius 2, 2048 bits) at three
similarity thresholds. 0.65 is the primary clustering view; 0.55 and 0.75 are
reported alongside it so the chemotype picture is not an artefact of one cut.

Seed selection is deliberately not "the top N molecules": it spreads across
distinct clusters, distinct Bemis-Murcko scaffolds, both PAR2 mechanisms, and a
mixture of near-reference, scaffold-hop, and novel chemistry.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

THRESHOLDS = [0.55, 0.65, 0.75]
PRIMARY_THRESHOLD = 0.65
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def build_fingerprints(smiles_list: list[str]):
    fps, valid = [], []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            continue
        fps.append(MORGAN.GetFingerprint(mol))
        valid.append(i)
    return fps, valid


def butina_clusters(fps, threshold: float) -> list[tuple[int, ...]]:
    """Butina clustering. RDKit expects a distance cutoff, so 1 - similarity."""
    n = len(fps)
    distances: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(distances, n, 1.0 - threshold, isDistData=True)
    return sorted(clusters, key=len, reverse=True)


def centroid_of(indices: list[int], fps) -> int:
    """Member with the highest mean similarity to the rest of its cluster."""
    if len(indices) == 1:
        return indices[0]
    best, best_score = indices[0], -1.0
    for i in indices:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in indices if j != i])
        score = float(np.mean(sims))
        if score > best_score:
            best, best_score = i, score
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postdock", type=Path, required=True,
                        help="DENOVO_POSTDOCK_ALL.csv or TOP500")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=400,
                        help="how many top-ranked designs to cluster")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--max-seeds-per-cluster", type=int, default=1)
    parser.add_argument("--min-pocket-fraction", type=float, default=0.25,
                        help="floor on each pocket's share of the seeds; not a fixed quota")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.postdock).sort_values("postdock_rank").head(args.top_n).reset_index(drop=True)
    fps, valid = build_fingerprints(table.smiles_prepared.tolist())
    table = table.iloc[valid].reset_index(drop=True)
    print(f"[cluster] clustering {len(table)} designs", flush=True)

    assignments: dict[float, list[int]] = {}
    cluster_sets: dict[float, list[tuple[int, ...]]] = {}
    for threshold in THRESHOLDS:
        clusters = butina_clusters(fps, threshold)
        cluster_sets[threshold] = clusters
        labels = [-1] * len(table)
        for cid, members in enumerate(clusters, 1):
            for m in members:
                labels[m] = cid
        assignments[threshold] = labels
        table[f"cluster_at_{threshold:.2f}"] = labels
        print(f"[cluster] threshold {threshold:.2f}: {len(clusters)} clusters, "
              f"largest {len(clusters[0]) if clusters else 0}", flush=True)

    primary_clusters = cluster_sets[PRIMARY_THRESHOLD]
    table["cluster_id"] = table[f"cluster_at_{PRIMARY_THRESHOLD:.2f}"]

    summary_rows = []
    for cid, members in enumerate(primary_clusters, 1):
        members = list(members)
        sub = table.iloc[members]
        centroid_idx = centroid_of(members, fps)
        best_idx = int(sub.postdock_rank.idxmin())
        pockets = sub.pocket.value_counts()
        summary_rows.append(dict(
            cluster_id=cid,
            n_members=len(members),
            pocket_majority=pockets.index[0],
            pocket_mix=";".join(f"{k}:{v}" for k, v in pockets.items()),
            family_mix=";".join(f"{k}:{v}" for k, v in sub.denovo_family.value_counts().items()),
            centroid_cand_id=table.cand_id.iloc[centroid_idx],
            centroid_smiles=table.smiles_prepared.iloc[centroid_idx],
            best_member_cand_id=table.cand_id.iloc[best_idx],
            best_member_smiles=table.smiles_prepared.iloc[best_idx],
            best_postdock_rank=int(sub.postdock_rank.min()),
            median_postdock_rank=float(sub.postdock_rank.median()),
            best_postdock_score=float(sub.postdock_score.max()),
            median_postdock_score=float(sub.postdock_score.median()),
            best_ifp_recovery=float(sub.ifp_recovery.max()) if sub.ifp_recovery.notna().any() else None,
            median_ifp_recovery=float(sub.ifp_recovery.median()) if sub.ifp_recovery.notna().any() else None,
            best_affinity=float(sub.best_affinity.min()) if sub.best_affinity.notna().any() else None,
            median_affinity=float(sub.best_affinity.median()) if sub.best_affinity.notna().any() else None,
            best_CNN_VS=float(sub.best_CNN_VS.max()) if sub.best_CNN_VS.notna().any() else None,
            median_CNN_VS=float(sub.best_CNN_VS.median()) if sub.best_CNN_VS.notna().any() else None,
            median_ligand_efficiency=float(sub.ligand_efficiency.median())
            if sub.ligand_efficiency.notna().any() else None,
            murcko_scaffold=table.murcko.iloc[centroid_idx],
            n_distinct_scaffolds=int(sub.murcko.nunique()),
            mean_mw=float(sub.mw.mean()), mean_clogp=float(sub.clogp.mean()),
            mean_tpsa=float(sub.tpsa.mean()), mean_fsp3=float(sub.fsp3.mean()),
            mean_qed=float(sub.qed.mean()),
            reference_antagonist=sub.reference_antagonist.iloc[0],
            median_tanimoto_to_reference=float(sub.tanimoto_to_reference.median()),
            # Convergence: several independently generated high-ranking designs
            # landing on the same chemotype is weak but real corroboration.
            n_members_in_top100=int((sub.postdock_rank <= 100).sum()),
            convergent_chemotype=bool((sub.postdock_rank <= 100).sum() >= 3),
            cluster_at_0_55=int(table[f"cluster_at_0.55"].iloc[centroid_idx]),
            cluster_at_0_75=int(table[f"cluster_at_0.75"].iloc[centroid_idx]),
        ))
    summary = pd.DataFrame(summary_rows).sort_values("best_postdock_rank").reset_index(drop=True)
    summary.to_csv(out / "DENOVO_CLUSTER_SUMMARY.csv", index=False)
    table.to_csv(out / "DENOVO_CLUSTER_MEMBERS.csv", index=False)

    # Seed selection: walk clusters best-first, respecting scaffold and mechanism spread.
    seeds: list[dict] = []
    used_scaffolds: dict[str, int] = {}
    per_cluster: dict[int, int] = {}
    pocket_counts: dict[str, int] = {}

    def novelty_band(tanimoto: float) -> str:
        if tanimoto >= 0.40:
            return "close_to_reference_antagonist"
        if tanimoto >= 0.20:
            return "moderate_scaffold_hop"
        return "novel_chemotype"

    ordered = table.sort_values("postdock_rank")
    pockets_present = list(ordered.pocket.unique())
    floor_per_pocket = int(np.ceil(args.n_seeds * args.min_pocket_fraction)) if pockets_present else 0

    def eligible(row) -> bool:
        cid = int(row.cluster_id)
        if per_cluster.get(cid, 0) >= args.max_seeds_per_cluster:
            return False
        scaffold = row.murcko if isinstance(row.murcko, str) else ""
        return not (scaffold and used_scaffolds.get(scaffold, 0) >= 1)

    def take(row) -> None:
        cid = int(row.cluster_id)
        scaffold = row.murcko if isinstance(row.murcko, str) else ""
        tanimoto = float(row.tanimoto_to_reference) if pd.notna(row.tanimoto_to_reference) else 0.0
        seeds.append(dict(
            seed_id="",
            cand_id=row.cand_id,
            family=row.denovo_family,
            pocket=row.pocket,
            canonical_smiles=row.smiles_prepared,
            postdock_rank=int(row.postdock_rank),
            postdock_score=float(row.postdock_score),
            cluster_id=cid,
            cluster_size=int(summary.loc[summary.cluster_id == cid, "n_members"].iloc[0]),
            convergent_chemotype=bool(
                summary.loc[summary.cluster_id == cid, "convergent_chemotype"].iloc[0]),
            best_affinity=row.best_affinity,
            best_CNNscore=getattr(row, "best_CNNscore", None),
            best_CNNaffinity=getattr(row, "best_CNNaffinity", None),
            best_CNN_VS=row.best_CNN_VS,
            ligand_efficiency=row.ligand_efficiency,
            ifp_recovery=row.ifp_recovery,
            reference_antagonist=row.reference_antagonist,
            Tanimoto_to_reference=row.tanimoto_to_reference,
            murcko_scaffold=scaffold,
            MW=row.mw, cLogP=row.clogp, TPSA=row.tpsa, fsp3=row.fsp3, QED=row.qed,
            novelty_band=novelty_band(tanimoto),
            selection_pass="",
        ))
        per_cluster[cid] = per_cluster.get(cid, 0) + 1
        pocket_counts[row.pocket] = pocket_counts.get(row.pocket, 0) + 1
        if scaffold:
            used_scaffolds[scaffold] = used_scaffolds.get(scaffold, 0) + 1

    # Pass 1: guarantee each pocket a floor of seeds, best-ranked first.
    for pocket in pockets_present:
        for row in ordered[ordered.pocket == pocket].itertuples():
            if pocket_counts.get(pocket, 0) >= floor_per_pocket or len(seeds) >= args.n_seeds:
                break
            if eligible(row):
                take(row)
                seeds[-1]["selection_pass"] = "pocket_floor"

    # Pass 2: fill the rest strictly by post-docking rank, pocket-blind.
    taken = {s["cand_id"] for s in seeds}
    for row in ordered.itertuples():
        if len(seeds) >= args.n_seeds:
            break
        if row.cand_id in taken or not eligible(row):
            continue
        take(row)
        seeds[-1]["selection_pass"] = "rank"

    for i, seed in enumerate(seeds, 1):
        seed["seed_id"] = f"SEED{i:02d}"

    seed_frame = pd.DataFrame(seeds)
    if not seed_frame.empty:
        seed_frame["reason_selected"] = seed_frame.apply(
            lambda r: (f"rank {r.postdock_rank} in {r.pocket}; cluster {r.cluster_id} "
                       f"(size {r.cluster_size}"
                       f"{', convergent chemotype' if r.convergent_chemotype else ''}); "
                       f"distinct Murcko scaffold; {r.novelty_band}; "
                       f"selected on {r.selection_pass}"), axis=1)
        seed_frame.to_csv(out / "SIMILARITY_SEARCH_SEEDS.csv", index=False)

    info = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        designs_clustered=int(len(table)),
        fingerprint="Morgan/ECFP4 radius 2, 2048 bits",
        clusters_by_threshold={f"{t:.2f}": len(cluster_sets[t]) for t in THRESHOLDS},
        singletons_by_threshold={f"{t:.2f}": int(sum(1 for c in cluster_sets[t] if len(c) == 1))
                                 for t in THRESHOLDS},
        primary_threshold=PRIMARY_THRESHOLD,
        largest_primary_cluster=int(len(primary_clusters[0])) if primary_clusters else 0,
        convergent_chemotypes=int(summary.convergent_chemotype.sum()),
        seeds_selected=int(len(seed_frame)),
        seeds_by_pocket={k: int(v) for k, v in seed_frame.pocket.value_counts().items()}
        if not seed_frame.empty else {},
        seeds_by_family={k: int(v) for k, v in seed_frame.family.value_counts().items()}
        if not seed_frame.empty else {},
        seeds_by_selection_pass={k: int(v) for k, v in seed_frame.selection_pass.value_counts().items()}
        if not seed_frame.empty else {},
        min_pocket_fraction=args.min_pocket_fraction,
        seeds_by_novelty_band={k: int(v) for k, v in seed_frame.novelty_band.value_counts().items()}
        if not seed_frame.empty else {},
        seed_distinct_clusters=int(seed_frame.cluster_id.nunique()) if not seed_frame.empty else 0,
        seed_distinct_scaffolds=int(seed_frame.murcko_scaffold.nunique()) if not seed_frame.empty else 0,
    )
    (out / "DENOVO_CLUSTER_SUMMARY.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
