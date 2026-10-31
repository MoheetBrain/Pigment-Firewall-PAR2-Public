#!/usr/bin/env python3
"""Verify the published snapshot without running research experiments."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'PUBLICATION_MANIFEST.json').read_text())
failures=[]
for item in manifest['files']:
    path=root/item['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        failures.append(item['path'])
if failures:
    raise SystemExit('Hash verification failed: '+', '.join(failures))
print(f"Verified {len(manifest['files'])} published files; no research computation run.")
