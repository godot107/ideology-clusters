"""Fetch Congress.gov topics for every bill voted on in a Congress's House roll calls (cached).

    python scripts/fetch_bills.py 119      # ~3 requests per bill; a few minutes the first time
"""
import sys

from ideology import congress_gov, voteview

congress = int(sys.argv[1]) if len(sys.argv) > 1 else 119
rc = voteview.rollcalls(congress, "H")
bills = sorted({b for b in rc.bill_number.dropna() if congress_gov.parse_bill(b)})
print(f"{len(bills)} bills on {len(rc)} roll calls; {rc.bill_number.isna().sum()} roll calls have no bill number")
missing = 0
for i, b in enumerate(bills, 1):
    r = congress_gov.bill(congress, b)
    missing += bool(r and r.get("missing"))
    if i % 50 == 0:
        print(i, flush=True)
print("done; not found on Congress.gov:", missing)
