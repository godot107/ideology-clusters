"""Caucus membership: labels the votes never saw, used to check whether clusters mean anything.

Source: English Wikipedia's current-member lists (119th Congress), fetched at runtime with the
retrieval date recorded. Caveats, reported with every result:
- the Freedom Caucus does not publish its roster; Wikipedia lists *publicly identified* members;
- lists change; a member who joined or left mid-Congress is labelled by the list on the fetch date.

Matching is by state + district (unambiguous within one Congress), then checked by surname so a
seat that changed hands mid-term isn't credited to the wrong person.
"""
import html as htmlmod
import io
import json
import re
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from ideology import voteview

DIR = voteview.DATA_DIR.parent / "caucus"
PAGES = {  # caucus: (Wikipedia page, section index of the current House member list)
    "Freedom Caucus": ("Freedom_Caucus", 30),
    "Progressive Caucus": ("Congressional_Progressive_Caucus", 17),
    "Blue Dog Coalition": ("Blue_Dog_Coalition", 11),
    "Problem Solvers Caucus": ("Problem_Solvers_Caucus", 9),
}
STATES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
          "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
          "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
          "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
          "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
          "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
          "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
          "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
          "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
          "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}


def fetch(page, section):
    path = DIR / f"{page}.json"
    if not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        url = (f"https://en.wikipedia.org/w/api.php?action=parse&page={page}&section={section}"
               "&prop=text&format=json&formatversion=2")
        req = urllib.request.Request(url, headers={"User-Agent": voteview.USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        data["retrieved"] = date.today().isoformat()
        path.write_text(json.dumps(data))
    data = json.loads(path.read_text())
    return data["parse"]["text"], data.get("retrieved")


def _text(fragment):
    return htmlmod.unescape(re.sub(r"<[^>]+>", "", fragment)).replace("‑", "-").strip()


def _seat(code):
    """'AZ-5' / 'D-CA-24' / 'AK-AL' -> ('AZ', 5) / ('CA', 24) / ('AK', 1)"""
    m = re.search(r"([A-Z]{2})-(\d+|AL)\b", code)
    if not m:
        return None
    return m.group(1), 1 if m.group(2) == "AL" else int(m.group(2))


def members(caucus):
    page, section = PAGES[caucus]
    html, retrieved = fetch(page, section)
    rows = []
    if caucus in ("Progressive Caucus", "Blue Dog Coalition"):
        t = pd.read_html(io.StringIO(html))[0]
        for _, r in t.iterrows():
            seat = _seat(str(r["District"]))
            if seat:
                rows.append({"name": str(r["Member"]), "state": seat[0], "district": seat[1]})
    else:
        if caucus == "Problem Solvers Caucus":   # stop before the former-members part
            html = re.split(r'id="Former', html)[0]
        for li in re.findall(r"<li>(.*?)</li>", html, flags=re.S):
            txt = _text(li)
            seat = _seat(txt)
            if seat:
                rows.append({"name": re.split(r"\s*\(", txt)[0], "state": seat[0], "district": seat[1]})
    df = pd.DataFrame(rows).drop_duplicates(["state", "district"])
    df["caucus"], df["retrieved"] = caucus, retrieved
    return df


SUFFIXES = {"JR", "SR", "II", "III", "IV"}


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z ]", " ", s)


def label(info):
    """info: members frame (from matrix.with_members). Returns (labels per member, match report)."""
    lab = pd.Series("none", index=info.index, dtype=object)
    report = {}
    for caucus in PAGES:
        cm = members(caucus)
        matched, unmatched = 0, []
        for _, r in cm.iterrows():
            cand = info[(info.state_abbrev == r.state) & (info.district_code == r.district)]
            surname = [w for w in _norm(r["name"]).split() if w not in SUFFIXES][-1]
            cand = cand[cand.bioname.map(_norm).str.contains(rf"\b{surname}\b", regex=True)]
            if len(cand) == 1:
                i = cand.index[0]
                lab[i] = caucus if lab[i] == "none" else lab[i] + " + " + caucus
                matched += 1
            else:
                unmatched.append(r["name"])
        report[caucus] = {"listed": len(cm), "matched": matched, "unmatched": unmatched,
                          "retrieved": cm.retrieved.iloc[0] if len(cm) else None}
    return lab, report
