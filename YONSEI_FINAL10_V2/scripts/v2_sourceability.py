#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2_sourceability.py -- reusable, cached, rate-limited compound sourceability resolver.

PURPOSE
-------
Turn "can we actually buy this molecule?" from a guess into a data field with
provenance.  Given a list of structures (isomeric SMILES + InChIKey) this tool
queries public chemical registries and vendor-catalogue indexes and emits, per
molecule per hit:

    molecule_id, vendor, catalogue_id, smiles, inchikey, match_type,
    stock_status, make_on_demand_status, price, quantity, purity, lead_time,
    country_or_source, source_url, source_endpoint, retrieved_date

and classifies each molecule into exactly one of four classes.

CLASSIFICATION DEFINITIONS (authoritative -- these are the definitions used by
the code in ``classify_molecule``):

  IN_STOCK
      A NAMED VENDOR lists a SPECIFIC CATALOGUE ID together with an affirmative
      in-stock / available status THAT THIS TOOL ACTUALLY RETRIEVED.  In
      practice this means either (a) an affirmative availability statement was
      read from a vendor product page / vendor API response, or (b) the exact
      structure was found in a vendor IN-STOCK catalogue snapshot whose
      snapshot date is recorded in ``catalogue_snapshot``.  Because (b) is a
      snapshot and not a live stock query, every IN_STOCK row carries
      ``stock_evidence`` and ``catalogue_snapshot`` so the staleness is visible.
      An IN_STOCK classification is only awarded on an EXACT (full 27-character
      InChIKey) match.  A connectivity-only match never yields IN_STOCK.

  MAKE_ON_DEMAND
      The structure is a member of a synthesise-on-demand chemical space (e.g.
      Enamine REAL, ZINC on-demand, Mcule Ultimate/Virtual) with a catalogue or
      space identifier, but no physical stock is asserted.

  REGISTERED_NOT_VERIFIED
      The structure exists in a chemical registry (PubChem / ChEMBL / UniChem)
      and may carry supplier annotations, but NO purchasable catalogue entry
      could be confirmed.  PubChem "Chemical Vendors" annotations land here by
      default: a PubChem vendor annotation is a LEAD TO VERIFY, not stock
      confirmation.

  NOT_SOURCEABLE
      Every configured resolver was run and returned no hit of any kind -- the
      structure is absent from PubChem, ChEMBL, UniChem, Mcule and every Arthor
      catalogue index that was queried.  The ``classification_reason`` field
      records which resolvers were run so the negative is auditable.  This is
      the expected result for de-novo generated structures.

MATCHING RULES
--------------
Matching is done on InChIKey, never on name.

  EXACT              full 27-character InChIKey identical to the query
                     (query key, or the key of the neutralised parent).
  CONNECTIVITY_ONLY  first 14 characters (skeleton block) identical but the
                     full key differs -- i.e. SAME SKELETON, DIFFERENT
                     STEREOCHEMISTRY / ISOTOPE / PROTONATION STATE.
                     *** THIS IS NOT THE SAME COMPOUND. ***
                     Our candidates carry defined stereochemistry; ordering a
                     connectivity-only match buys the wrong stereoisomer and
                     wastes an assay slot.  Such rows are always flagged and
                     are never allowed to produce an IN_STOCK classification.
  NO_MATCH           neither.

Ligand-prep SMILES in this project are protonated (e.g. ``[NH2+]``), so their
InChIKeys end in ``-O``/``-N`` suffixes that will NOT match registry entries for
the neutral parent.  Every query is therefore resolved twice: as supplied, and
as the neutralised, desalted parent (RDKit ``rdMolStandardize``).  Both keys are
recorded.

DATA SOURCES ACTUALLY USED (all public, no credentials)
------------------------------------------------------
  PubChem PUG-REST   https://pubchem.ncbi.nlm.nih.gov/rest/pug/...
      - /compound/inchikey/{key}/cids/TXT            exact structure -> CID
      - /compound/fastidentity/smiles/{smi}/cids/TXT?identity_type=same_connectivity
                                                     connectivity-only siblings
      - /compound/cid/{cid}/property/...             structure + name readback
  PubChem PUG-View   /rest/pug_view/categories/compound/{cid}/JSON
      - "Chemical Vendors" category: vendor SourceName, RegistryID (catalogue
        number) and SourceRecordURL (direct product page).  LEADS, not stock.
  UniChem v1         https://www.ebi.ac.uk/unichem/api/v1/compounds
      - InChIKey -> cross-references.  Source 28 is MolPort (a vendor
        aggregator) and yields real ``Molport-xxx-xxx-xxx`` catalogue IDs.
        Source 34 (DrugCentral, an approved-drugs-only resource) is recorded
        as APPROVAL evidence.  Sources 2/46/14 (DrugBank / ClinicalTrials /
        FDA SRS) are recorded only as regulatory REGISTRY CROSS-REFERENCES in
        ``regulatory_registry_crossrefs`` -- DrugBank covers experimental and
        investigational compounds and an FDA SRS UNII is merely a substance
        identifier, so NEITHER implies approval and neither is allowed into
        ``approved_drug_evidence``.
  ChEMBL             https://www.ebi.ac.uk/chembl/api/data/molecule.json
      - exact InChIKey lookup.  ``max_phase`` is captured and translated into
        a plain-language ``clinical_phase_note``; ONLY max_phase 4 counts as
        approval.  max_phase 3 means "reached Phase 3, NOT approved".
  Mcule              https://mcule.com/api/v1/search/lookup/?query={smiles}
      - anonymous; returns MCULE-xxxxxxxxxx id + product URL.
        /api/v1/compound/{id}/ returns the InChIKey for exact verification.
        NOTE: /prices/ and /availability/ require credentials (HTTP 401), and
        the anonymous lookup endpoint returns HTTP 429 under sustained use --
        429 is retried (honouring Retry-After) and is NEVER cached.
  Arthor             https://arthor.docking.org/dt/{db}/search?query=&type=Substructure
      - UCSF Shoichet-lab public catalogue index.  Substructure search with the
        whole molecule as query returns supersets; the exact molecule is always
        among them if present.  Hits are re-matched locally on InChIKey.
        Databases queried are listed in ``ARTHOR_DATABASES`` below and include
        Enamine REAL (make-on-demand), ZINC in-stock, ChemSpace screening
        stock, Mcule in-stock/virtual, and DrugBank approved-drug sets.

SOURCES INVESTIGATED AND BLOCKED (recorded, not silently dropped)
----------------------------------------------------------------
  MolPort API   api.molport.com/api/chemical-search/search -> HTTP 200 body
                {"Result":{"Status":2,"Message":"Username or password is
                incorrect!"}}.  Requires an account we do not have.
  Mcule pricing mcule.com/api/v1/compound/{id}/prices/ and /availability/
                -> HTTP 401 {"detail":"Authentication credentials were not
                provided."}.  Requires an API token we do not have.
                The anonymous search/lookup endpoint also rate-limits with
                HTTP 429; that is throttling, NOT a negative result, so it is
                retried and never cached (see _write_cache guard).
  Enamine REAL  real.enamine.net API requires an Enamine REAL Explorer account
                and an X-API-KEY.  The REAL database bulk download is ~13.6
                billion molecules and is not tractable here.  Enamine REAL
                membership is therefore established via the Arthor
                REAL-Database index instead.
  Chemspace API api.chem-space.com requires an API key.
  ZINC web      zinc20.docking.org / zinc15.docking.org redirect to /captcha.
                Deliberately NOT bypassed.  ZINC content is reached instead via
                the Arthor and SmallWorld indexes, which are not captcha-gated.
  eMolecules    no public unauthenticated search API found.
  MedChemExpress
                medchemexpress.com returns HTTP 403 to programmatic access, so
                any MCE catalogue number reported elsewhere is UNVERIFIED here.

NO VALUE IS EVER INVENTED.  Fields that could not be retrieved are the literal
string ``UNKNOWN`` (not 0, not a plausible placeholder).  ``price``,
``quantity``, ``purity`` and ``lead_time`` are UNKNOWN for essentially every
row because every route that carries them is credential-gated -- that is the
honest state of the world, and it is reported rather than filled in.

USAGE
-----
  python v2_sourceability.py \
      --input  path/to/candidates.csv \
      --id-col cand_id --smiles-col smiles_prepared --inchikey-col inchikey \
      --outdir OUTDIR --cache-dir OUTDIR/cache --tag TOP20

  # several inputs into one master table
  python v2_sourceability.py --input a.csv --input b.csv ... --outdir OUT

Outputs written to --outdir:
  <tag>_SOURCEABILITY_MASTER.csv     one row per molecule per vendor/registry hit
  <tag>_SOURCEABILITY_SUMMARY.csv    one row per molecule, best classification
  <tag>_SOURCEABILITY_PROVENANCE.json endpoints, counts, cache location, dates

Reruns are cheap: every HTTP response is cached to disk keyed by the exact
request, so re-running on a different input list only hits the network for
molecules not seen before.  Use --refresh to force re-fetch.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# RDKit (required for standardisation and InChIKey computation)
# ----------------------------------------------------------------------------
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi as rd_inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception as _e:  # pragma: no cover
    _HAVE_RDKIT = False
    _RDKIT_ERR = str(_e)

UNKNOWN = "UNKNOWN"
NOT_VERIFIED = "NOT_VERIFIED"

USER_AGENT = (
    "PAR2-sourceability/1.0 (academic target-discovery pipeline; "
    "public catalogue lookup only)"
)

# ----------------------------------------------------------------------------
# Arthor catalogue databases.
#
# 'kind' drives classification:
#     in_stock       -> physical stock asserted by the catalogue snapshot
#     on_demand      -> synthesise-on-demand chemical space
#     for_sale       -> purchasable but stock vs on-demand not distinguished
#     approved_drug  -> regulatory/approval evidence, not a purchase route
# 'snapshot' is the catalogue vintage encoded in the database name; it is
# reported verbatim so staleness is visible to the reader.
# ----------------------------------------------------------------------------
ARTHOR_DATABASES: List[Dict[str, str]] = [
    {"db": "MMcule-In-Stock-25Q3-6.5M", "vendor": "Mcule",
     "kind": "in_stock", "snapshot": "2025-Q3"},
    {"db": "ChemSpace-SC-Stock-Mar2022-346K", "vendor": "Chemspace",
     "kind": "in_stock", "snapshot": "2022-03"},
    {"db": "In-Stock-19Q4-14.1M", "vendor": "ZINC in-stock aggregate (multi-vendor)",
     "kind": "in_stock", "snapshot": "2019-Q4"},
    {"db": "REAL-Database-22Q1", "vendor": "Enamine REAL",
     "kind": "on_demand", "snapshot": "2022-Q1"},
    {"db": "ZINC-On-Demand-19Q4-311M", "vendor": "ZINC on-demand aggregate",
     "kind": "on_demand", "snapshot": "2019-Q4"},
    {"db": "Mcule-Ultimate-25Q3-111M", "vendor": "Mcule Ultimate (virtual)",
     "kind": "on_demand", "snapshot": "2025-Q3"},
    {"db": "ZINC20-ForSale-22Q1", "vendor": "ZINC20 for-sale aggregate",
     "kind": "for_sale", "snapshot": "2022-Q1"},
    {"db": "DB-world-approved-2506-2.6K", "vendor": "DrugBank world-approved set",
     "kind": "approved_drug", "snapshot": "2025-06"},
    {"db": "DB-US-approved-2506-1.9K", "vendor": "DrugBank US-approved set",
     "kind": "approved_drug", "snapshot": "2025-06"},
]

# UniChem source ids we care about.
UNICHEM_VENDOR_SOURCES = {28: "MolPort"}
UNICHEM_STATUS_SOURCES = {
    2: "DrugBank",
    34: "DrugCentral",
    46: "ClinicalTrials",
    14: "FDA/USP Substance Registration System (SRS)",
    4: "Guide to Pharmacology",
}

MASTER_FIELDS = [
    "molecule_id", "input_file", "query_smiles", "query_inchikey",
    "neutral_smiles", "neutral_inchikey", "skeleton14",
    "resolver", "record_class",
    "vendor", "catalogue_id", "hit_smiles", "hit_inchikey", "match_type",
    "stock_status", "make_on_demand_status", "stock_evidence",
    "catalogue_snapshot",
    "price", "quantity", "purity", "lead_time", "country_or_source",
    "source_url", "source_endpoint", "retrieved_date", "notes",
]

SUMMARY_FIELDS = [
    "molecule_id", "input_file", "query_smiles", "query_inchikey",
    "neutral_smiles", "neutral_inchikey", "skeleton14",
    "classification", "classification_reason",
    "best_vendor", "best_catalogue_id", "best_source_url",
    "exact_match_found", "connectivity_only_match_found",
    "n_exact_hits", "n_connectivity_only_hits", "n_vendor_leads",
    "pubchem_cid_exact", "pubchem_cid_connectivity", "chembl_id",
    "chembl_max_phase", "approved_drug_evidence",
    "regulatory_registry_crossrefs", "clinical_phase_note",
    "vendor_leads_pubchem", "molport_catalogue_ids", "mcule_ids",
    "enamine_real_member", "in_stock_catalogues", "on_demand_catalogues",
    "price", "quantity", "purity", "lead_time",
    "resolvers_run", "resolvers_failed", "retrieved_date",
]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


# ============================================================================
# HTTP with disk cache + per-host rate limiting
# ============================================================================
class HttpClient:
    """Cached, rate-limited HTTP GET/POST.

    Every response (including error responses) is written to the cache so a
    rerun costs nothing and so the provenance of a negative result is as
    auditable as a positive one.
    """

    # Some hosts drop connections under sustained request bursts.  Arthor runs
    # multi-second substructure scans and needs a wider gap than PubChem.
    # Measured 2026-09-06: arthor.docking.org serves 3 concurrent connections
    # reliably (9 database queries in 1.8 s) but drops connections at 5.  So it
    # is driven with a semaphore of 3 rather than a strict serial interval,
    # which is ~10x faster than serialising the nine catalogue queries.
    HOST_CONCURRENCY = {"arthor.docking.org": 3}

    HOST_INTERVAL = {
        "arthor.docking.org": 0.15,
        "pubchem.ncbi.nlm.nih.gov": 0.34,
        "www.ebi.ac.uk": 0.34,
        "mcule.com": 1.5,
    }

    def __init__(self, cache_dir: str, min_interval: float = 0.34,
                 refresh: bool = False, timeout: int = 90,
                 max_retries: int = 4, verbose: bool = True):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.min_interval = min_interval
        self.refresh = refresh
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self._last_call: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._sems: Dict[str, threading.Semaphore] = {
            h: threading.Semaphore(n) for h, n in self.HOST_CONCURRENCY.items()}
        # Circuit breaker: after this many consecutive throttling responses a
        # host is considered rate-limited for the rest of the run.  Further
        # requests to it return immediately with status 429 and error
        # "circuit_open", so a large input list degrades to an honest,
        # explicitly-recorded coverage gap instead of hours of retry backoff.
        self.throttle_streak: Dict[str, int] = Counter()
        self.circuit_open: Dict[str, str] = {}
        self.stats = Counter()
        self.endpoints_hit: Dict[str, int] = Counter()

    # -- cache -------------------------------------------------------------
    def _key(self, method: str, url: str, body: Optional[str]) -> str:
        raw = f"{method}\n{url}\n{body or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _write_cache(self, key: str, rec: Dict[str, Any]) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        os.replace(tmp, p)

    # -- throttle ----------------------------------------------------------
    def _throttle(self, url: str) -> None:
        host = urllib.parse.urlparse(url).netloc
        interval = self.HOST_INTERVAL.get(host, self.min_interval)
        with self._lock:
            last = self._last_call.get(host, 0.0)
            wait = interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.time()

    # -- fetch -------------------------------------------------------------
    def fetch(self, url: str, method: str = "GET", body: Optional[str] = None,
              headers: Optional[Dict[str, str]] = None,
              accept_status: Iterable[int] = (200,)) -> Dict[str, Any]:
        """Return {'status','text','url','retrieved_utc','from_cache','error'}."""
        key = self._key(method, url, body)
        cached = None if self.refresh else self._read_cache(key)
        if cached is not None:
            self.stats["cache_hit"] += 1
            cached["from_cache"] = True
            return cached

        host = urllib.parse.urlparse(url).netloc
        if host in self.circuit_open:
            self.stats["circuit_open_skipped"] += 1
            return {"status": 429, "text": "", "url": url, "method": method,
                    "body": body, "retrieved_utc": utc_now(),
                    "from_cache": False,
                    "error": f"circuit_open: {self.circuit_open[host]}"}

        hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if headers:
            hdrs.update(headers)

        rec: Dict[str, Any] = {
            "status": 0, "text": "", "url": url, "method": method,
            "body": body, "retrieved_utc": utc_now(),
            "from_cache": False, "error": "",
        }

        sem = self._sems.get(urllib.parse.urlparse(url).netloc)
        for attempt in range(1, self.max_retries + 1):
            self._throttle(url)
            if sem is not None:
                sem.acquire()
            try:
                data = body.encode("utf-8") if body is not None else None
                req = urllib.request.Request(url, data=data, headers=hdrs,
                                             method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    rec["status"] = resp.status
                    rec["text"] = resp.read().decode("utf-8", "replace")
                    rec["url"] = resp.geturl()
                    rec["retrieved_utc"] = utc_now()
                break
            except urllib.error.HTTPError as e:
                rec["status"] = e.code
                try:
                    rec["text"] = e.read().decode("utf-8", "replace")
                except Exception:
                    rec["text"] = ""
                rec["retrieved_utc"] = utc_now()
                # 404 from PubChem/ChEMBL is a legitimate "not found" answer;
                # 401/403 mean credentials are required.  Those are final.
                if e.code in (400, 404, 401, 403):
                    break
                if attempt == self.max_retries:
                    rec["error"] = f"HTTPError {e.code}"
                elif e.code in (408, 425, 429):
                    # Throttled.  Honour Retry-After when the server sends one,
                    # otherwise back off hard -- this is not a miss.
                    try:
                        wait = float(e.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        wait = 0.0
                    time.sleep(max(wait, 5.0 * attempt))
                else:
                    time.sleep(1.5 * attempt)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["retrieved_utc"] = utc_now()
                if attempt == self.max_retries:
                    break
                time.sleep(1.5 * attempt)
            finally:
                if sem is not None:
                    sem.release()

        with self._lock:
            if rec["status"] in (408, 425, 429):
                self.throttle_streak[host] += 1
                if (self.throttle_streak[host] >= 5
                        and host not in self.circuit_open):
                    self.circuit_open[host] = (
                        f"{self.throttle_streak[host]} consecutive throttling "
                        f"responses; host rate-limited, stopped querying it at "
                        f"{utc_now()}")
                    print(f"    [circuit-open] {host} is rate-limiting; "
                          f"remaining requests to it will be recorded as "
                          f"coverage gaps rather than retried",
                          file=sys.stderr)
            elif 200 <= rec["status"] < 400:
                self.throttle_streak[host] = 0

        self.stats["network"] += 1
        self.stats[f"status_{rec['status']}"] += 1
        base = url.split("?")[0]
        self.endpoints_hit[base] += 1
        # Only cache DEFINITIVE answers.  A transport failure (status 0), a
        # 5xx, or a throttling response (408/425/429) is not an answer --
        # caching it would make a transient blip permanently indistinguishable
        # from a real negative result, and a rerun would silently inherit the
        # gap.  Those are left uncached so the next run retries them.
        # (Learned the hard way: Mcule's anonymous API starts returning 429
        # under sustained use, and 429 < 500, so an earlier version of this
        # check cached seven rate-limit responses as if they were real misses.)
        if 200 <= rec["status"] < 500 and rec["status"] not in (408, 425, 429):
            self._write_cache(key, rec)
        else:
            self.stats["uncached_transient_failure"] += 1
        if self.verbose and rec["status"] not in accept_status:
            print(f"    [http {rec['status']}] {url[:120]} {rec['error']}",
                  file=sys.stderr)
        return rec

    def json(self, url: str, **kw) -> Tuple[Optional[Any], Dict[str, Any]]:
        rec = self.fetch(url, **kw)
        if rec["status"] != 200 or not rec["text"].strip():
            return None, rec
        try:
            return json.loads(rec["text"]), rec
        except Exception:
            return None, rec


# ============================================================================
# Structure standardisation
# ============================================================================
class Structure:
    """A query structure with both the as-supplied and neutralised parent form."""

    __slots__ = ("mol_id", "input_file", "raw_smiles", "raw_inchikey",
                 "smiles", "inchikey", "neutral_smiles", "neutral_inchikey",
                 "skeletons", "keys", "error")

    def __init__(self, mol_id: str, raw_smiles: str, raw_inchikey: str,
                 input_file: str = ""):
        self.mol_id = mol_id
        self.input_file = input_file
        self.raw_smiles = (raw_smiles or "").strip()
        self.raw_inchikey = (raw_inchikey or "").strip()
        self.smiles = self.raw_smiles
        self.inchikey = self.raw_inchikey
        self.neutral_smiles = ""
        self.neutral_inchikey = ""
        self.error = ""
        self._standardise()
        self.keys = {k for k in (self.inchikey, self.neutral_inchikey) if k}
        self.skeletons = {k.split("-")[0] for k in self.keys if len(k) >= 14}

    def _standardise(self) -> None:
        if not _HAVE_RDKIT:
            self.error = "rdkit_unavailable"
            return
        if not self.raw_smiles:
            self.error = "no_smiles"
            return
        mol = Chem.MolFromSmiles(self.raw_smiles)
        if mol is None:
            self.error = "smiles_parse_failed"
            return
        try:
            self.smiles = Chem.MolToSmiles(mol)
            k = rd_inchi.MolToInchiKey(mol)
            if k:
                self.inchikey = k
        except Exception as e:
            self.error = f"inchikey_failed:{e}"

        # Neutral, largest-fragment parent -- registries store the parent.
        try:
            frag = rdMolStandardize.LargestFragmentChooser().choose(mol)
            neu = rdMolStandardize.Uncharger().uncharge(frag)
            neu = rdMolStandardize.Normalize(neu)
            Chem.SanitizeMol(neu)
            self.neutral_smiles = Chem.MolToSmiles(neu)
            self.neutral_inchikey = rd_inchi.MolToInchiKey(neu) or ""
        except Exception as e:
            self.neutral_smiles = self.smiles
            self.neutral_inchikey = self.inchikey
            if not self.error:
                self.error = f"neutralise_failed:{e}"

    @property
    def skeleton14(self) -> str:
        return (self.inchikey or self.neutral_inchikey or "").split("-")[0]

    def match(self, other_key: str) -> str:
        """EXACT / CONNECTIVITY_ONLY / NO_MATCH against this query."""
        if not other_key:
            return "NO_MATCH"
        other_key = other_key.strip().upper()
        if other_key in self.keys:
            return "EXACT"
        if other_key.split("-")[0] in self.skeletons:
            return "CONNECTIVITY_ONLY"
        return "NO_MATCH"

    def match_smiles(self, smi: str) -> Tuple[str, str]:
        """Compute the hit InChIKey from SMILES and classify. -> (match, key)"""
        if not _HAVE_RDKIT or not smi:
            return "NO_MATCH", ""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return "NO_MATCH", ""
        try:
            k = rd_inchi.MolToInchiKey(m) or ""
        except Exception:
            return "NO_MATCH", ""
        return self.match(k), k


# ============================================================================
# Hit record
# ============================================================================
def make_hit(struct: Structure, resolver: str, record_class: str, **kw) -> Dict[str, str]:
    row = {f: "" for f in MASTER_FIELDS}
    row.update({
        "molecule_id": struct.mol_id,
        "input_file": struct.input_file,
        "query_smiles": struct.smiles,
        "query_inchikey": struct.inchikey,
        "neutral_smiles": struct.neutral_smiles,
        "neutral_inchikey": struct.neutral_inchikey,
        "skeleton14": struct.skeleton14,
        "resolver": resolver,
        "record_class": record_class,
        "vendor": UNKNOWN, "catalogue_id": UNKNOWN,
        "hit_smiles": "", "hit_inchikey": "", "match_type": UNKNOWN,
        "stock_status": UNKNOWN, "make_on_demand_status": UNKNOWN,
        "stock_evidence": UNKNOWN, "catalogue_snapshot": UNKNOWN,
        "price": UNKNOWN, "quantity": UNKNOWN, "purity": UNKNOWN,
        "lead_time": UNKNOWN, "country_or_source": UNKNOWN,
        "source_url": "", "source_endpoint": "",
        "retrieved_date": today(), "notes": "",
    })
    row.update({k: v for k, v in kw.items() if k in row})
    return row


# ============================================================================
# Resolvers
# ============================================================================
class Resolver:
    """Bundle of source-specific lookups.  Each returns a list of master rows."""

    PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
    UNICHEM = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
    CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    MCULE = "https://mcule.com/api/v1"
    ARTHOR = "https://arthor.docking.org/dt"

    def __init__(self, http: HttpClient, arthor_dbs: Optional[List[Dict]] = None,
                 skip: Optional[set] = None, arthor_limit: int = 200,
                 verbose: bool = True):
        self.http = http
        self.arthor_dbs = arthor_dbs if arthor_dbs is not None else ARTHOR_DATABASES
        self.skip = skip or set()
        self.arthor_limit = arthor_limit
        self.verbose = verbose
        self.failures: Dict[str, List[str]] = defaultdict(list)

    # ---------------------------------------------------------------- PubChem
    def pubchem(self, s: Structure) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        exact_cids: List[str] = []

        # 1. exact InChIKey -> CID, for each key form we hold
        for key in [k for k in (s.inchikey, s.neutral_inchikey) if k]:
            url = f"{self.PUBCHEM}/compound/inchikey/{urllib.parse.quote(key)}/cids/TXT"
            rec = self.http.fetch(url, accept_status=(200, 404))
            if rec["status"] == 200:
                cids = [c for c in rec["text"].split() if c.strip().isdigit()]
                for cid in cids:
                    if cid in exact_cids:
                        continue
                    exact_cids.append(cid)
                    rows.append(make_hit(
                        s, "pubchem_exact", "REGISTRY",
                        vendor="PubChem (registry, not a vendor)",
                        catalogue_id=f"CID{cid}",
                        hit_inchikey=key, match_type="EXACT",
                        stock_status=NOT_VERIFIED,
                        make_on_demand_status=NOT_VERIFIED,
                        country_or_source="NCBI PubChem",
                        source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                        source_endpoint=url,
                        retrieved_date=rec["retrieved_utc"][:10],
                        notes=f"exact InChIKey {key} resolved to CID"))
            elif rec["status"] not in (200, 404):
                self.failures[s.mol_id].append(f"pubchem_exact:{rec['status']}")

        # 2. connectivity-only siblings via fastidentity
        smi = s.neutral_smiles or s.smiles
        if smi:
            url = (f"{self.PUBCHEM}/compound/fastidentity/smiles/"
                   f"{urllib.parse.quote(smi, safe='')}/cids/TXT"
                   f"?identity_type=same_connectivity")
            rec = self.http.fetch(url, accept_status=(200, 404))
            if rec["status"] == 200:
                sib = [c for c in rec["text"].split()
                       if c.strip().isdigit() and c not in exact_cids][:25]
                if sib:
                    rows.append(make_hit(
                        s, "pubchem_connectivity", "REGISTRY_CONNECTIVITY",
                        vendor="PubChem (registry, not a vendor)",
                        catalogue_id=";".join(f"CID{c}" for c in sib),
                        match_type="CONNECTIVITY_ONLY",
                        stock_status=NOT_VERIFIED,
                        make_on_demand_status=NOT_VERIFIED,
                        country_or_source="NCBI PubChem",
                        source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{sib[0]}",
                        source_endpoint=url,
                        retrieved_date=rec["retrieved_utc"][:10],
                        notes=("same-connectivity CIDs -- DIFFERENT stereochemistry "
                               "and/or protonation; NOT the same compound")))
            elif rec["status"] not in (200, 404):
                self.failures[s.mol_id].append(f"pubchem_conn:{rec['status']}")

        # 3. vendor annotations for each exact CID (leads only)
        for cid in exact_cids[:3]:
            rows.extend(self._pubchem_vendors(s, cid))
        return rows

    def _pubchem_vendors(self, s: Structure, cid: str) -> List[Dict[str, str]]:
        url = f"{self.PUBCHEM_VIEW}/categories/compound/{cid}/JSON"
        data, rec = self.http.json(url, accept_status=(200, 404))
        rows: List[Dict[str, str]] = []
        if not data:
            return rows
        cats = (data.get("SourceCategories") or {}).get("Categories") or []
        for cat in cats:
            if cat.get("Category") != "Chemical Vendors":
                continue
            for src in (cat.get("Sources") or []):
                name = src.get("SourceName") or UNKNOWN
                reg = src.get("RegistryID") or UNKNOWN
                # A chemical name in the RegistryID slot is not a catalogue id.
                if reg != UNKNOWN and (len(reg) > 60 or " " in reg.strip()):
                    reg_note = "vendor supplied a chemical name, not a catalogue number"
                    reg = UNKNOWN
                else:
                    reg_note = ""
                rows.append(make_hit(
                    s, "pubchem_vendor_annotation", "VENDOR_LEAD",
                    vendor=name, catalogue_id=reg,
                    match_type="EXACT",
                    stock_status=NOT_VERIFIED,
                    make_on_demand_status=NOT_VERIFIED,
                    stock_evidence=("PubChem Chemical Vendors annotation -- "
                                    "supplier LEAD, no stock statement retrieved"),
                    country_or_source=src.get("SourceURL") or UNKNOWN,
                    source_url=(src.get("SourceRecordURL")
                                or src.get("SourceDetail") or ""),
                    source_endpoint=url,
                    retrieved_date=rec["retrieved_utc"][:10],
                    notes=("PubChem SID %s. %s" % (src.get("SID"), reg_note)).strip()))
        return rows

    # ---------------------------------------------------------------- UniChem
    def unichem(self, s: Structure) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for key in [k for k in (s.inchikey, s.neutral_inchikey) if k]:
            body = json.dumps({"type": "inchikey", "compound": key})
            data, rec = self.http.json(
                self.UNICHEM, method="POST", body=body,
                headers={"Content-Type": "application/json"},
                accept_status=(200, 404))
            if not data:
                if rec["status"] not in (200, 404):
                    self.failures[s.mol_id].append(f"unichem:{rec['status']}")
                continue
            for comp in (data.get("compounds") or []):
                for src in (comp.get("sources") or []):
                    sid = src.get("id")
                    cid_ = src.get("compoundId") or UNKNOWN
                    if sid in UNICHEM_VENDOR_SOURCES:
                        rows.append(make_hit(
                            s, "unichem_vendor", "VENDOR_CATALOGUE_ID",
                            vendor=UNICHEM_VENDOR_SOURCES[sid],
                            catalogue_id=cid_,
                            hit_inchikey=key, match_type="EXACT",
                            stock_status=NOT_VERIFIED,
                            make_on_demand_status=NOT_VERIFIED,
                            stock_evidence=("UniChem cross-reference to a MolPort "
                                            "shop entry -- catalogue listing "
                                            "confirmed, live stock NOT confirmed "
                                            "(MolPort API needs credentials)"),
                            country_or_source="MolPort (Latvia) aggregator",
                            source_url=src.get("url") or "",
                            source_endpoint=self.UNICHEM,
                            retrieved_date=rec["retrieved_utc"][:10],
                            notes=f"UniChem srcID {sid}, query key {key}"))
                    elif sid in UNICHEM_STATUS_SOURCES:
                        rows.append(make_hit(
                            s, "unichem_status", "STATUS_EVIDENCE",
                            vendor=UNICHEM_STATUS_SOURCES[sid],
                            catalogue_id=cid_,
                            hit_inchikey=key, match_type="EXACT",
                            stock_status=NOT_VERIFIED,
                            make_on_demand_status=NOT_VERIFIED,
                            country_or_source=UNICHEM_STATUS_SOURCES[sid],
                            source_url=src.get("url") or "",
                            source_endpoint=self.UNICHEM,
                            retrieved_date=rec["retrieved_utc"][:10],
                            notes=("clinical/regulatory registry cross-reference, "
                                   "not a purchase route")))
        return rows

    # ----------------------------------------------------------------- ChEMBL
    def chembl(self, s: Structure) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for key in [k for k in (s.inchikey, s.neutral_inchikey) if k]:
            url = (f"{self.CHEMBL}?molecule_structures__standard_inchi_key="
                   f"{urllib.parse.quote(key)}")
            data, rec = self.http.json(url, accept_status=(200, 404))
            if not data:
                if rec["status"] not in (200, 404):
                    self.failures[s.mol_id].append(f"chembl:{rec['status']}")
                continue
            for m in (data.get("molecules") or []):
                chid = m.get("molecule_chembl_id") or UNKNOWN
                phase = m.get("max_phase")
                phase_s = UNKNOWN if phase in (None, "") else str(phase)
                rows.append(make_hit(
                    s, "chembl", "STATUS_EVIDENCE",
                    vendor="ChEMBL (registry, not a vendor)",
                    catalogue_id=chid,
                    hit_inchikey=key, match_type="EXACT",
                    stock_status=NOT_VERIFIED,
                    make_on_demand_status=NOT_VERIFIED,
                    country_or_source="EMBL-EBI ChEMBL",
                    source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chid}/",
                    source_endpoint=url,
                    retrieved_date=rec["retrieved_utc"][:10],
                    notes=(f"max_phase={phase_s}; pref_name="
                           f"{m.get('pref_name') or UNKNOWN}")))
        return rows

    # ------------------------------------------------------------------ Mcule
    def mcule(self, s: Structure) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        smi = s.neutral_smiles or s.smiles
        if not smi:
            return rows
        url = f"{self.MCULE}/search/lookup/?query={urllib.parse.quote(smi, safe='')}"
        data, rec = self.http.json(url, accept_status=(200, 404))
        if not data:
            if rec["status"] not in (200, 404):
                self.failures[s.mol_id].append(f"mcule:{rec['status']}")
            return rows
        for r in (data.get("results") or [])[:10]:
            mid = r.get("mcule_id") or UNKNOWN
            hit_smi = r.get("smiles") or ""
            match, hitkey = s.match_smiles(hit_smi)
            # Mcule's lookup returns similar compounds, most of which are not
            # ours.  Only spend a second network call confirming the ones that
            # could still match: if RDKit already derived a key from Mcule's own
            # SMILES and its skeleton block differs from the query's, the
            # compounds are constitutionally different and no InChIKey nuance
            # will change that.  If RDKit could NOT derive a key (empty hitkey),
            # fall through and ask Mcule -- never skip on a parse failure.
            if match == "NO_MATCH" and hitkey:
                continue
            # Confirm with mcule's own InChIKey where available.
            if mid != UNKNOWN:
                d2, r2 = self.http.json(f"{self.MCULE}/compound/{mid}/",
                                        accept_status=(200, 404))
                if d2 and d2.get("inchi_key"):
                    hitkey = d2["inchi_key"]
                    match = s.match(hitkey)
            if match == "NO_MATCH":
                continue
            rows.append(make_hit(
                s, "mcule_lookup",
                "VENDOR_CATALOGUE_ID" if match == "EXACT" else "CONNECTIVITY_LEAD",
                vendor="Mcule", catalogue_id=mid,
                hit_smiles=hit_smi, hit_inchikey=hitkey, match_type=match,
                stock_status=NOT_VERIFIED,
                make_on_demand_status=NOT_VERIFIED,
                stock_evidence=("Mcule anonymous lookup API returned a catalogue "
                                "id; /prices/ and /availability/ are HTTP 401 "
                                "without an API token, so stock and price were "
                                "NOT retrieved"),
                country_or_source="Mcule (Hungary) aggregator",
                source_url=r.get("url") or f"https://mcule.com/{mid}/",
                source_endpoint=url,
                retrieved_date=rec["retrieved_utc"][:10],
                notes=("EXACT structure match" if match == "EXACT"
                       else "SKELETON ONLY -- different stereochemistry")))
        return rows

    # ----------------------------------------------------------------- Arthor
    def arthor(self, s: Structure) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        smi = s.neutral_smiles or s.smiles
        if not smi:
            return rows
        q = urllib.parse.quote(smi, safe="")

        # Fetch all catalogue databases concurrently -- the HttpClient caps
        # arthor.docking.org at 3 simultaneous connections (see HOST_CONCURRENCY),
        # which is what the server tolerates.  Responses are then processed
        # serially, because the InChIKey re-matching is the part that must be
        # deterministic.
        def _fetch(spec):
            url = (f"{self.ARTHOR}/{urllib.parse.quote(spec['db'])}/search"
                   f"?query={q}&type=Substructure&length={self.arthor_limit}")
            data, rec = self.http.json(url, accept_status=(200, 400, 404))
            return spec, url, data, rec

        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            fetched = list(pool.map(_fetch, self.arthor_dbs))

        for spec, url, data, rec in fetched:
            db = spec["db"]
            if not data:
                if rec["status"] not in (200, 400, 404):
                    self.failures[s.mol_id].append(f"arthor[{db}]:{rec['status']}")
                continue
            if data.get("error"):
                self.failures[s.mol_id].append(f"arthor[{db}]:{data['error']}")
                continue
            for entry in (data.get("data") or []):
                # [rank, index, SMILES, Identifier, source]
                if len(entry) < 4:
                    continue
                hit_smi, ident = entry[2], entry[3]
                match, hitkey = s.match_smiles(hit_smi)
                if match == "NO_MATCH":
                    continue  # a genuine superset, not our molecule
                kind = spec["kind"]
                if kind == "in_stock":
                    stock = ("IN_STOCK_PER_CATALOGUE_SNAPSHOT"
                             if match == "EXACT" else NOT_VERIFIED)
                    ondem = "N/A"
                    ev = (f"exact structure present in vendor IN-STOCK catalogue "
                          f"snapshot '{db}' (vintage {spec['snapshot']}); this is a "
                          f"SNAPSHOT, not a live stock query")
                    rc = "IN_STOCK_CATALOGUE"
                elif kind == "on_demand":
                    stock = "NO_PHYSICAL_STOCK"
                    ondem = ("MAKE_ON_DEMAND_SPACE_MEMBER"
                             if match == "EXACT" else NOT_VERIFIED)
                    ev = (f"exact structure is a member of synthesise-on-demand "
                          f"space '{db}' (vintage {spec['snapshot']})")
                    rc = "ON_DEMAND_CATALOGUE"
                elif kind == "approved_drug":
                    stock = NOT_VERIFIED
                    ondem = NOT_VERIFIED
                    ev = (f"structure present in approved-drug reference set "
                          f"'{db}' -- regulatory evidence, not a purchase route")
                    rc = "APPROVED_DRUG_EVIDENCE"
                else:
                    stock = NOT_VERIFIED
                    ondem = NOT_VERIFIED
                    ev = (f"structure present in for-sale index '{db}' "
                          f"(vintage {spec['snapshot']}); stock vs on-demand "
                          f"not distinguished by this index")
                    rc = "FOR_SALE_CATALOGUE"
                if match != "EXACT":
                    rc = "CONNECTIVITY_LEAD"
                rows.append(make_hit(
                    s, f"arthor:{db}", rc,
                    vendor=spec["vendor"], catalogue_id=ident,
                    hit_smiles=hit_smi, hit_inchikey=hitkey, match_type=match,
                    stock_status=stock, make_on_demand_status=ondem,
                    stock_evidence=ev, catalogue_snapshot=spec["snapshot"],
                    country_or_source="Arthor index, UCSF (arthor.docking.org)",
                    source_url=(f"https://arthor.docking.org/search.html"
                                f"?db={urllib.parse.quote(db)}"
                                f"&query={q}&type=Substructure"),
                    source_endpoint=url,
                    retrieved_date=rec["retrieved_utc"][:10],
                    notes=("EXACT structure match" if match == "EXACT"
                           else "SKELETON ONLY -- different stereochemistry")))
        return rows

    # ------------------------------------------------------------------ drive
    ALL = ("pubchem", "unichem", "chembl", "mcule", "arthor")

    def resolve(self, s: Structure) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        rows: List[Dict[str, str]] = []
        run, failed = [], []
        for name in self.ALL:
            if name in self.skip:
                continue
            try:
                rows.extend(getattr(self, name)(s))
                run.append(name)
            except Exception as e:
                failed.append(f"{name}:{type(e).__name__}:{e}")
                if self.verbose:
                    print(f"    [resolver-error] {s.mol_id} {name}: {e}",
                          file=sys.stderr)
        failed.extend(self.failures.get(s.mol_id, []))
        return rows, run, failed


# ============================================================================
# Classification
# ============================================================================
def classify_molecule(s: Structure, rows: List[Dict[str, str]],
                      run: List[str], failed: List[str]) -> Dict[str, str]:
    exact = [r for r in rows if r["match_type"] == "EXACT"]
    conn = [r for r in rows if r["match_type"] == "CONNECTIVITY_ONLY"]

    in_stock = [r for r in exact
                if r["stock_status"] == "IN_STOCK_PER_CATALOGUE_SNAPSHOT"]
    on_dem = [r for r in exact
              if r["make_on_demand_status"] == "MAKE_ON_DEMAND_SPACE_MEMBER"]
    for_sale = [r for r in exact if r["record_class"] == "FOR_SALE_CATALOGUE"]
    vendor_ids = [r for r in exact if r["record_class"] == "VENDOR_CATALOGUE_ID"]
    leads = [r for r in exact if r["record_class"] == "VENDOR_LEAD"]
    approved = [r for r in exact if r["record_class"] == "APPROVED_DRUG_EVIDENCE"]
    registry = [r for r in exact if r["record_class"].startswith("REGISTRY")
                or r["record_class"] == "STATUS_EVIDENCE"]

    # ChEMBL max_phase 4 is itself approval evidence.
    chembl_rows = [r for r in exact if r["resolver"] == "chembl"]
    chembl_id, max_phase = UNKNOWN, UNKNOWN
    if chembl_rows:
        chembl_id = chembl_rows[0]["catalogue_id"]
        m = re.search(r"max_phase=([^;]+)", chembl_rows[0]["notes"] or "")
        if m:
            max_phase = m.group(1).strip()

    # Approval evidence must mean APPROVAL, not merely "present in a database
    # that also contains approved drugs".  DrugBank covers experimental and
    # investigational compounds, and an FDA SRS UNII is only a substance
    # identifier -- neither implies approval.  Those go in a separate
    # cross-reference column so they cannot be misread as regulatory status.
    approved_evidence = []
    registry_xrefs = []
    if approved:  # membership of the DrugBank *approved* subsets via Arthor
        approved_evidence += sorted({r["vendor"] for r in approved})
    try:
        if max_phase not in (UNKNOWN, "") and float(max_phase) >= 4:
            approved_evidence.append("ChEMBL max_phase=4 (approved)")
    except ValueError:
        pass
    for r in exact:
        # DrugCentral is an approved-drugs-only resource, so it IS evidence.
        if r["vendor"] == "DrugCentral":
            approved_evidence.append(f"DrugCentral:{r['catalogue_id']}")
        elif r["vendor"] in ("DrugBank",
                             "FDA/USP Substance Registration System (SRS)",
                             "ClinicalTrials", "Guide to Pharmacology"):
            registry_xrefs.append(f"{r['vendor']}:{r['catalogue_id']}")

    # Human-readable reading of ChEMBL max_phase, so nobody has to guess what
    # "3.0" means or infer approval from a bare number.
    _PHASE = {"4": "APPROVED (ChEMBL max_phase 4)",
              "3": "reached Phase 3, NOT approved",
              "2": "reached Phase 2, NOT approved",
              "1": "reached Phase 1, NOT approved",
              "0": "preclinical / no clinical phase recorded",
              "-1": "preclinical / unknown phase"}
    try:
        phase_note = _PHASE.get(str(int(float(max_phase))), UNKNOWN) \
            if max_phase not in (UNKNOWN, "", "None") else UNKNOWN
    except (ValueError, TypeError):
        phase_note = UNKNOWN

    # ---- the four classes -------------------------------------------------
    if in_stock:
        cls = "IN_STOCK"
        best = in_stock[0]
        reason = ("exact-InChIKey member of a vendor in-stock catalogue snapshot: "
                  + "; ".join(sorted({f"{r['vendor']}={r['catalogue_id']} "
                                      f"({r['catalogue_snapshot']})"
                                      for r in in_stock})))
    elif on_dem:
        cls = "MAKE_ON_DEMAND"
        best = on_dem[0]
        reason = ("exact-InChIKey member of a synthesise-on-demand space: "
                  + "; ".join(sorted({f"{r['vendor']}={r['catalogue_id']}"
                                      for r in on_dem})))
    elif for_sale or vendor_ids:
        # Listed for sale / has a vendor catalogue id, but no affirmative stock
        # statement was retrieved.  By the stated definitions this is NOT
        # IN_STOCK; it is a confirmed catalogue listing awaiting verification.
        cls = "REGISTERED_NOT_VERIFIED"
        best = (vendor_ids or for_sale)[0]
        reason = ("vendor catalogue listing confirmed but NO affirmative stock "
                  "status retrievable without credentials: "
                  + "; ".join(sorted({f"{r['vendor']}={r['catalogue_id']}"
                                      for r in (vendor_ids + for_sale)})[:6]))
    elif leads or registry or exact:
        cls = "REGISTERED_NOT_VERIFIED"
        best = (leads or registry or exact)[0]
        if leads:
            reason = ("present in registry with PubChem supplier annotations "
                      "(LEADS, unverified): "
                      + "; ".join(sorted({r["vendor"] for r in leads})[:8]))
        else:
            reason = ("present in chemical registry (PubChem/ChEMBL/UniChem) but "
                      "no vendor catalogue entry found in any queried index")
    elif conn:
        cls = "NOT_SOURCEABLE"
        best = None
        reason = ("no exact-structure hit in any queried source. "
                  f"{len(conn)} CONNECTIVITY-ONLY hit(s) exist (same skeleton, "
                  "different stereochemistry/protonation) -- these are DIFFERENT "
                  "compounds and do not make this molecule sourceable. "
                  f"Resolvers run: {','.join(run)}")
    else:
        cls = "NOT_SOURCEABLE"
        best = None
        reason = ("determined by exhaustive negative: exact-InChIKey and "
                  "connectivity searches returned zero hits across "
                  f"{','.join(run)} (PubChem exact + same-connectivity, ChEMBL, "
                  "UniChem, Mcule lookup, and every Arthor catalogue index "
                  "queried). Expected for de-novo generated structures.")

    if failed:
        reason += f" | resolver issues: {';'.join(sorted(set(failed))[:5])}"

    def joinset(rs, field):
        vals = sorted({r[field] for r in rs if r[field] and r[field] != UNKNOWN})
        return ";".join(vals[:12]) if vals else UNKNOWN

    pc_exact = [r for r in exact if r["resolver"] == "pubchem_exact"]
    pc_conn = [r for r in rows if r["resolver"] == "pubchem_connectivity"]
    molport = [r for r in exact if r["vendor"] == "MolPort"]
    mcule_r = [r for r in exact if r["vendor"] == "Mcule"
               and r["resolver"] == "mcule_lookup"]
    real = [r for r in on_dem if "Enamine REAL" in r["vendor"]]

    return {
        "molecule_id": s.mol_id,
        "input_file": s.input_file,
        "query_smiles": s.smiles,
        "query_inchikey": s.inchikey,
        "neutral_smiles": s.neutral_smiles,
        "neutral_inchikey": s.neutral_inchikey,
        "skeleton14": s.skeleton14,
        "classification": cls,
        "classification_reason": reason,
        "best_vendor": best["vendor"] if best else UNKNOWN,
        "best_catalogue_id": best["catalogue_id"] if best else UNKNOWN,
        "best_source_url": best["source_url"] if best else UNKNOWN,
        "exact_match_found": "YES" if exact else "NO",
        "connectivity_only_match_found": "YES" if conn else "NO",
        "n_exact_hits": str(len(exact)),
        "n_connectivity_only_hits": str(len(conn)),
        "n_vendor_leads": str(len(leads)),
        "pubchem_cid_exact": joinset(pc_exact, "catalogue_id"),
        "pubchem_cid_connectivity": joinset(pc_conn, "catalogue_id"),
        "chembl_id": chembl_id,
        "chembl_max_phase": max_phase,
        "approved_drug_evidence": (";".join(sorted(set(approved_evidence)))
                                   if approved_evidence else "NONE_FOUND"),
        "regulatory_registry_crossrefs": (";".join(sorted(set(registry_xrefs)))
                                          if registry_xrefs else "NONE_FOUND"),
        "clinical_phase_note": phase_note,
        "vendor_leads_pubchem": joinset(leads, "vendor"),
        "molport_catalogue_ids": joinset(molport, "catalogue_id"),
        "mcule_ids": joinset(mcule_r, "catalogue_id"),
        "enamine_real_member": "YES" if real else "NO",
        "in_stock_catalogues": joinset(in_stock, "vendor"),
        "on_demand_catalogues": joinset(on_dem, "vendor"),
        # Never invented: every price route is credential-gated.
        "price": UNKNOWN,
        "quantity": UNKNOWN,
        "purity": UNKNOWN,
        "lead_time": UNKNOWN,
        "resolvers_run": ",".join(run),
        "resolvers_failed": ";".join(sorted(set(failed))) if failed else "",
        "retrieved_date": today(),
    }


# ============================================================================
# IO
# ============================================================================
def read_input(path: str, id_col: str, smiles_col: str,
               inchikey_col: str) -> List[Structure]:
    out: List[Structure] = []
    base = os.path.basename(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []

        def pick(pref: str, cands: List[str]) -> str:
            if pref in cols:
                return pref
            for c in cands:
                if c in cols:
                    return c
            return ""

        idc = pick(id_col, ["cand_id", "molecule_id", "id", "compound_id",
                            "name", "compound"])
        smc = pick(smiles_col, ["smiles_prepared", "smiles", "isomeric_smiles",
                                "canonical_smiles", "SMILES"])
        ikc = pick(inchikey_col, ["inchikey", "InChIKey", "inchi_key"])
        if not smc:
            raise SystemExit(f"{path}: no SMILES column found (looked for "
                             f"{smiles_col!r}); columns are {cols}")
        for i, row in enumerate(rdr):
            mid = (row.get(idc) or f"{base}:{i+1}").strip()
            out.append(Structure(mid, row.get(smc, ""),
                                 row.get(ikc, "") if ikc else "", base))
    return out


def write_csv(path: str, fields: List[str], rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


# ============================================================================
# main
# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Resolve compound sourceability with full provenance.")
    ap.add_argument("--input", action="append", required=True,
                    help="input CSV (repeatable)")
    ap.add_argument("--id-col", default="cand_id")
    ap.add_argument("--smiles-col", default="smiles_prepared")
    ap.add_argument("--inchikey-col", default="inchikey")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cache and re-fetch everything")
    ap.add_argument("--skip", default="",
                    help="comma-separated resolvers to skip "
                         "(pubchem,unichem,chembl,mcule,arthor)")
    ap.add_argument("--arthor-limit", type=int, default=200)
    ap.add_argument("--arthor-dbs", default="",
                    help="comma-separated Arthor database names to query "
                         "(default: all of ARTHOR_DATABASES). Use to keep large "
                         "lists tractable, e.g. "
                         "--arthor-dbs In-Stock-19Q4-14.1M,REAL-Database-22Q1,"
                         "MMcule-In-Stock-25Q3-6.5M . Whatever subset is used is "
                         "recorded in the provenance JSON so coverage stays honest.")
    ap.add_argument("--list-arthor-dbs", action="store_true",
                    help="print the configured Arthor databases and exit")
    ap.add_argument("--min-interval", type=float, default=0.34,
                    help="minimum seconds between calls to the same host")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N only")
    args = ap.parse_args(argv)

    if args.list_arthor_dbs:
        for d in ARTHOR_DATABASES:
            print(f"{d['db']:<36} {d['kind']:<14} {d['snapshot']:<10} {d['vendor']}")
        return 0

    if not _HAVE_RDKIT:
        print(f"FATAL: rdkit unavailable ({_RDKIT_ERR}). Structure "
              f"standardisation and InChIKey matching are mandatory.",
              file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.outdir, "cache")
    http = HttpClient(cache_dir, min_interval=args.min_interval,
                      refresh=args.refresh)
    adbs = None
    if args.arthor_dbs.strip():
        want = {x.strip() for x in args.arthor_dbs.split(",") if x.strip()}
        adbs = [d for d in ARTHOR_DATABASES if d["db"] in want]
        missing = want - {d["db"] for d in adbs}
        if missing:
            raise SystemExit(f"unknown Arthor database(s): {sorted(missing)}. "
                             f"Run --list-arthor-dbs to see valid names.")
        print(f"[sourceability] Arthor restricted to {len(adbs)} of "
              f"{len(ARTHOR_DATABASES)} databases: {sorted(want)}")
    res = Resolver(http, arthor_dbs=adbs,
                   skip={x.strip() for x in args.skip.split(",") if x.strip()},
                   arthor_limit=args.arthor_limit)

    structs: List[Structure] = []
    for p in args.input:
        structs.extend(read_input(p, args.id_col, args.smiles_col,
                                  args.inchikey_col))
    if args.limit:
        structs = structs[:args.limit]

    print(f"[sourceability] {len(structs)} structures from "
          f"{len(args.input)} file(s); cache={cache_dir}")

    master: List[Dict[str, str]] = []
    summary: List[Dict[str, str]] = []
    t0 = time.time()
    for i, s in enumerate(structs, 1):
        if s.error:
            print(f"  [{i}/{len(structs)}] {s.mol_id}: STRUCTURE ERROR {s.error}")
        rows, run, failed = res.resolve(s)
        master.extend(rows)
        summ = classify_molecule(s, rows, run, failed)
        summary.append(summ)
        print(f"  [{i}/{len(structs)}] {s.mol_id:<12} {summ['classification']:<24}"
              f" exact={summ['n_exact_hits']:>3} conn={summ['n_connectivity_only_hits']:>3}"
              f" leads={summ['n_vendor_leads']:>3}")

    tag = (args.tag + "_") if args.tag else ""
    mpath = os.path.join(args.outdir, f"{tag}SOURCEABILITY_MASTER.csv")
    spath = os.path.join(args.outdir, f"{tag}SOURCEABILITY_SUMMARY.csv")
    ppath = os.path.join(args.outdir, f"{tag}SOURCEABILITY_PROVENANCE.json")
    write_csv(mpath, MASTER_FIELDS, master)
    write_csv(spath, SUMMARY_FIELDS, summary)

    prov = {
        "tool": "YONSEI_FINAL10_V2/scripts/v2_sourceability.py",
        "run_started_utc": _dt.datetime.fromtimestamp(
            t0, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_finished_utc": utc_now(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "inputs": args.input,
        "n_structures": len(structs),
        "n_master_rows": len(master),
        "cache_location": os.path.abspath(cache_dir),
        "http_stats": dict(http.stats),
        "hosts_rate_limited_circuit_open": dict(http.circuit_open),
        "endpoints_hit": dict(http.endpoints_hit),
        "arthor_databases_queried": res.arthor_dbs,
        "classification_counts": dict(Counter(
            r["classification"] for r in summary)),
        "match_type_counts": dict(Counter(r["match_type"] for r in master)),
        "resolver_counts": dict(Counter(r["resolver"] for r in master)),
        "structures_with_errors": [
            {"id": s.mol_id, "error": s.error} for s in structs if s.error],
        "blocked_sources": {
            "MolPort API": "api.molport.com/api/chemical-search/search returns "
                           "Status 2 'Username or password is incorrect!' "
                           "without credentials",
            "Mcule prices/availability": "HTTP 401 'Authentication credentials "
                                         "were not provided.'",
            "Enamine REAL API": "real.enamine.net requires an Enamine REAL "
                                "Explorer account + X-API-KEY",
            "Chemspace API": "api.chem-space.com requires an API key",
            "ZINC web UI": "zinc20/zinc15.docking.org redirect to /captcha "
                           "(not bypassed); reached via Arthor index instead",
            "eMolecules": "no public unauthenticated search API located",
        },
        "outputs": {"master": os.path.abspath(mpath),
                    "summary": os.path.abspath(spath)},
    }
    with open(ppath, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    print(f"\n[sourceability] wrote:\n  {mpath}\n  {spath}\n  {ppath}")
    print(f"[sourceability] classification counts: "
          f"{prov['classification_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
