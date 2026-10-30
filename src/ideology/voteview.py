"""Voteview roll-call data: the single source of truth for what we download and how it's coded.

Source: Lewis, Poole, Rosenthal, Boche, Rudkin & Sonnet, *Voteview: Congressional Roll-Call Votes
Database* (https://voteview.com). Free to use with citation; downloaded at runtime, never committed.

Files per Congress and chamber (H = House, S = Senate), e.g. congress 118, chamber "H":
- members:   who served, party, state, and their DW-NOMINATE scores (the known answer we compare to)
- votes:     one row per (roll call, member) with how they voted
- rollcalls: one row per roll call: date, question, bill
"""
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://voteview.com/static/data/out"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "voteview"
USER_AGENT = "ideology-clusters/0.1 (contact@willieman.com)"

# Voteview cast codes. 1-3 are kinds of yea, 4-6 kinds of nay; 7-9 present / not voting; 0 = not a member.
YEA, NAY = {1, 2, 3}, {4, 5, 6}

PARTIES = {100: "Democrat", 200: "Republican", 328: "Independent"}
CHAMBERS = {"H": "House", "S": "Senate"}


def url(kind, congress, chamber):
    return f"{BASE}/{kind}/{chamber}{congress:03d}_{kind}.csv"


def fetch(kind, congress, chamber):
    """Download one file if missing; returns its path. Validates it looks like Voteview CSV."""
    path = DATA_DIR / kind / f"{chamber}{congress:03d}_{kind}.csv"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url(kind, congress, chamber), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not body.startswith(b"congress,"):
        raise ValueError(f"unexpected content from {url(kind, congress, chamber)}: {body[:80]!r}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


def members(congress, chamber):
    df = pd.read_csv(fetch("members", congress, chamber))
    df = df[df.chamber == CHAMBERS[chamber]]   # the file also lists the President
    df["party"] = df.party_code.map(PARTIES).fillna("Other")
    return df.set_index("icpsr")


def votes(congress, chamber):
    return pd.read_csv(fetch("votes", congress, chamber),
                       usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"])


def rollcalls(congress, chamber):
    return pd.read_csv(fetch("rollcalls", congress, chamber)).set_index("rollnumber")


def vote_type(question):
    """Coarse kind of House roll call from its `vote_question` text. "suspension" = bills brought up
    under suspension of the rules: no amendments, two-thirds needed, so in practice bipartisan bills
    the majority leadership schedules."""
    q = str(question)
    if "Suspend the Rules" in q:
        return "suspension"
    if "Amendment" in q:
        return "amendment"
    if "Resolution" in q or "Previous Question" in q:
        return "rule"
    if "Passage" in q or "Concur" in q:
        return "passage"
    return "other"
