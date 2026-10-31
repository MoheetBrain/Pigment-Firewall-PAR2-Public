#!/usr/bin/env python
"""
v3_vendor_check.py — lean, timeout-hardened vendor resolution.

Written because v2_sourceability.py stalled three times on hanging requests
during long runs. This does less, but it always terminates: every HTTP call has
a hard timeout, every molecule is independent, and partial results are flushed
continuously so a stall costs one molecule rather than the whole run.

Resolution route (highest-yield first, per the Agent 5 method doc):
  1. InChIKey -> PubChem CID, exact (full 27-char key) and connectivity-only
     (first 14 chars). Connectivity-only is a DIFFERENT compound and is
     labelled as such - ordering one buys the wrong stereoisomer.
  2. PubChem PUG-View "Chemical Vendors" section -> count of named suppliers.
  3. The molecule is resolved twice, as supplied and as its neutralised
     largest-fragment parent, because prepared SMILES are protonated and their
     InChIKeys never match a registry entry for the neutral parent.

Nothing is invented. A field that could not be retrieved is UNKNOWN.
Vendor COUNT is evidence of a purchase route; it is not a live stock check and
it carries no price or lead time.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
PC = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PV = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
UA = "Mozilla/5.0 (research; PAR2 sourceability check)"
TIMEOUT = 20


def get_json(url: str, timeout: int = TIMEOUT):
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf8", "ignore")), None
    except HTTPError as e:
        return None, f"http {e.code}"
    except (URLError, TimeoutError, OSError) as e:
        return None, f"neterr {type(e).__name__}"
    except Exception as e:
        return None, f"err {type(e).__name__}"


def cids_for_key(key: str):
    d, err = get_json(f"{PC}/compound/inchikey/{quote(key)}/cids/JSON")
    if d and "IdentifierList" in d:
        return d["IdentifierList"].get("CID", []), None
    return [], err


def vendor_count(cid: int):
    d, err = get_json(f"{PV}/categories/compound/{cid}/JSON", timeout=25)
    if not d:
        return None, err
    n, names = 0, []
    try:
        for cat in d.get("SourceCategories", {}).get("Categories", []):
            if "vendor" in str(cat.get("Category", "")).lower():
                srcs = cat.get("Sources", [])
                n = len(srcs)
                names = [s.get("SourceName") for s in srcs[:8] if s.get("SourceName")]
    except Exception as e:
        return None, f"parse {type(e).__name__}"
    return {"n_vendors": n, "vendors": names}, None


def resolve(rec):
    cid_, smi = rec["cand_id"], rec["smiles"]
    out = {"cand_id": cid_, "smiles": smi, "query_inchikey": None,
           "neutral_inchikey": None, "exact_cid": None, "connectivity_cids": None,
           "match_type": "NO_MATCH", "n_vendors": None, "vendors": None,
           "classification": "NOT_SOURCEABLE", "notes": "", "errors": ""}
    errs = []
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        out["classification"] = "INVALID_STRUCTURE"; return out
    try:
        k = inchi.MolToInchiKey(m); out["query_inchikey"] = k
    except Exception:
        k = None
    try:
        nm = rdMolStandardize.Uncharger().uncharge(
            rdMolStandardize.LargestFragmentChooser().choose(m))
        nk = inchi.MolToInchiKey(nm); out["neutral_inchikey"] = nk
    except Exception:
        nk = None

    exact = []
    for kk in [x for x in (k, nk) if x]:
        c, e = cids_for_key(kk)
        if e: errs.append(f"{kk[:14]}:{e}")
        exact += c or []
        time.sleep(0.2)
    exact = sorted(set(exact))
    if exact:
        out["exact_cid"] = exact[0]; out["match_type"] = "EXACT"
    else:
        skel = (nk or k or "")[:14]
        if skel:
            c, e = cids_for_key(skel)
            if e: errs.append(f"skel:{e}")
            if c:
                out["connectivity_cids"] = ";".join(str(x) for x in c[:5])
                out["match_type"] = "CONNECTIVITY_ONLY"
                out["notes"] = ("skeleton match only - DIFFERENT stereochemistry, "
                                "ordering this buys a different compound")

    if out["exact_cid"]:
        v, e = vendor_count(out["exact_cid"])
        if e: errs.append(f"vendors:{e}")
        if v:
            out["n_vendors"] = v["n_vendors"]
            out["vendors"] = ";".join(v["vendors"])
            if v["n_vendors"] > 0:
                out["classification"] = "IN_STOCK_VENDOR_LISTED"
            else:
                out["classification"] = "REGISTERED_NOT_VERIFIED"
        else:
            out["classification"] = "REGISTERED_NOT_VERIFIED"
    elif out["match_type"] == "CONNECTIVITY_ONLY":
        out["classification"] = "REGISTERED_NOT_VERIFIED"
    out["errors"] = ";".join(errs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", default="cand_id")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    recs = [{"cand_id": str(r[args.id_col]), "smiles": r[args.smiles_col]}
            for _, r in df.iterrows()]
    fields = ["cand_id", "smiles", "query_inchikey", "neutral_inchikey",
              "exact_cid", "connectivity_cids", "match_type", "n_vendors",
              "vendors", "classification", "notes", "errors"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields); w.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(resolve, r): r for r in recs}
            for f in as_completed(futs):
                try:
                    w.writerow(f.result())
                except Exception as e:
                    w.writerow({"cand_id": futs[f]["cand_id"],
                                "classification": "RESOLVER_ERROR",
                                "errors": str(e)})
                done += 1
                fo.flush()
                if done % 10 == 0:
                    print(f"  {done}/{len(recs)}", flush=True)
    d = pd.read_csv(args.out)
    print(json.dumps({"resolved": int(len(d)),
                      "by_class": {k: int(v) for k, v in
                                   d.classification.value_counts().items()},
                      "by_match": {k: int(v) for k, v in
                                   d.match_type.value_counts(dropna=False).items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
