#!/usr/bin/env python3
"""Carve a screening shard set out of the prepared ligand library.

The preparation stage prepares everything once. This selects which prepared
ligands actually go to GNINA, without re-embedding anything: the 3D coordinates
written by v2_prepare_ligands.py are copied verbatim, so a screen set is always
byte-identical to the corresponding records in the full library.

Default scope is the de novo designs only. The full library is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "04_V2_SCREEN")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", default="DENOVO",
                        help="comma-separated source labels to keep (DENOVO, SOURCEABLE)")
    parser.add_argument("--cand-ids", type=Path, default=None,
                        help="optional CSV with a cand_id column; overrides --sources")
    parser.add_argument("--shard-size", type=int, default=250)
    args = parser.parse_args()

    screen_dir = args.screen_dir.resolve()
    out = args.out.resolve()
    shard_out = out / "shards"
    shard_out.mkdir(parents=True, exist_ok=True)

    ligands = pd.read_csv(screen_dir / "V2_LIGAND_TABLE.csv")
    if args.cand_ids and args.cand_ids.exists():
        keep_ids = set(pd.read_csv(args.cand_ids).cand_id)
        scope = f"cand_ids from {args.cand_ids.name}"
    else:
        sources = {s.strip().upper() for s in args.sources.split(",") if s.strip()}
        keep_ids = set(ligands[ligands.source.isin(sources)].cand_id)
        scope = f"sources={sorted(sources)}"
    if not keep_ids:
        raise SystemExit("selection is empty")

    buffers: dict[str, list[str]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    written: dict[str, int] = defaultdict(int)
    files: list[Path] = []

    def flush(pocket: str) -> None:
        if not buffers[pocket]:
            return
        path = shard_out / f"{pocket}_shard{counters[pocket]:04d}.sdf"
        path.write_text("".join(buffers[pocket]))
        files.append(path)
        buffers[pocket] = []
        counters[pocket] += 1

    for shard in sorted(p for p in (screen_dir / "shards").glob("*.sdf")
                        if not p.name.startswith(".")):
        pocket = shard.name.split("_shard")[0]
        for record in shard.read_text().split("$$$$\n"):
            if not record.strip():
                continue
            name = record.lstrip("\n").split("\n", 1)[0].strip()
            if name not in keep_ids:
                continue
            buffers[pocket].append(record + "$$$$\n")
            written[pocket] += 1
            if len(buffers[pocket]) >= args.shard_size:
                flush(pocket)
    for pocket in list(buffers):
        flush(pocket)

    selected = ligands[ligands.cand_id.isin(keep_ids)]
    selected.to_csv(out / "SCREEN_SET_LIGANDS.csv", index=False)

    manifest = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        source_library=str(screen_dir),
        scope=scope,
        ligands_selected=int(len(selected)),
        ligands_by_source={k: int(v) for k, v in selected.source.value_counts().items()},
        ligands_by_family={k: int(v) for k, v in selected.denovo_family.value_counts().items()}
        if "denovo_family" in selected else {},
        docking_jobs=int(sum(written.values())),
        docking_jobs_by_pocket={k: int(v) for k, v in written.items()},
        shard_count=len(files),
        shard_size=args.shard_size,
        shard_sha256={f.name: sha256(f) for f in files},
        note="Coordinates copied verbatim from the prepared library; nothing re-embedded.",
    )
    (out / "SCREEN_SET_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "shard_sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
