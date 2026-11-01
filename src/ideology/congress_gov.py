"""Bill topics from the Congress.gov API (Library of Congress): the official policy area (one per bill,
assigned by the Congressional Research Service), legislative subject terms, and the latest CRS summary.

Voteview roll calls carry a bill number (e.g. "HR7148", "HJRES42"); this module turns that into the
bill's topics. Responses are cached as JSON in data/congress/ (gitignored), one file per bill, so a
re-run makes no requests. The API key is read from the environment (CONGRESS_API_KEY, e.g. from a
gitignored .env) and is never printed or logged: error messages name the bill, not the URL.

Free key: https://api.congress.gov/sign-up/ (5,000 requests/hour).
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "congress"
BASE = "https://api.congress.gov/v3"
USER_AGENT = "ideology-clusters/0.1 (https://github.com/godot107/ideology-clusters; contact@willieman.com)"
BILL_RE = re.compile(r"^(HCONRES|HJRES|HRES|HR|SCONRES|SJRES|SRES|S)(\d+)$")


def parse_bill(bill_number):
    """'HJRES42' -> ('hjres', 42); None if it isn't a bill or resolution number (e.g. a nomination)."""
    m = BILL_RE.match(str(bill_number or "").strip().upper())
    return (m.group(1).lower(), int(m.group(2))) if m else None


def _key():
    key = os.environ.get("CONGRESS_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("CONGRESS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("CONGRESS_API_KEY not set (put it in .env at the project root)")
    return key


def _get(path, params=None, retries=3):
    q = urllib.parse.urlencode({"format": "json", "limit": 250, **(params or {}), "api_key": _key()})
    req = urllib.request.Request(f"{BASE}/{path}?{q}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"Congress.gov {path}: HTTP {e.code}") from None
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"Congress.gov {path}: network error") from None


def bill(congress, bill_number, pause=0.25):
    """Topics for one bill: {"type", "number", "title", "policy_area", "subjects", "summary"}, cached.
    Returns None for numbers that aren't bills (or that Congress.gov doesn't know)."""
    parsed = parse_bill(bill_number)
    if parsed is None:
        return None
    t, n = parsed
    f = CACHE / str(congress) / f"{t}{n}.json"
    if f.exists():
        return json.loads(f.read_text())
    base = f"bill/{congress}/{t}/{n}"
    info = _get(base)
    if info is None:
        out = {"type": t, "number": n, "missing": True}
    else:
        b = info["bill"]
        subj = _get(f"{base}/subjects") or {}
        summ = _get(f"{base}/summaries") or {}
        sums = sorted(summ.get("summaries", []), key=lambda s: s.get("updateDate", ""))
        out = {"type": t, "number": n, "title": b.get("title"),
               "policy_area": (b.get("policyArea") or {}).get("name"),
               "subjects": [s["name"] for s in subj.get("subjects", {}).get("legislativeSubjects", [])],
               "summary": sums[-1]["text"] if sums else None,
               "url": f"https://www.congress.gov/bill/{congress}th-congress/{_slug(t)}/{n}"}
        time.sleep(pause)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def _slug(t):
    return {"hr": "house-bill", "hres": "house-resolution", "hjres": "house-joint-resolution",
            "hconres": "house-concurrent-resolution", "s": "senate-bill", "sres": "senate-resolution",
            "sjres": "senate-joint-resolution", "sconres": "senate-concurrent-resolution"}[t]
