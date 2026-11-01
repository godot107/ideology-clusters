"""How your representative votes: a non-partisan look-up for the 119th House (2025-26), from public
roll-call votes. Reads the CSVs that scripts/issues.py writes to results/; no API keys, no network.

    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RES = Path(__file__).resolve().parents[1] / "results"
PARTY = {"D": "Democrat", "R": "Republican"}
COLOR = alt.Scale(domain=["Democrat", "Republican"], range=["#2f6fd0", "#d6453d"])
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia", "PR": "Puerto Rico", "GU": "Guam",
    "VI": "U.S. Virgin Islands", "AS": "American Samoa", "MP": "Northern Mariana Islands"}

st.set_page_config(page_title="How your representative votes", page_icon="🗳️", layout="wide")


@st.cache_data
def load():
    m = pd.read_csv(RES / "issues_members_119H.csv")
    m["party_name"] = m.party.map(PARTY)
    m["state_name"] = m.state.map(STATES).fillna(m.state)
    m["seat"] = m.state + "-" + m.district.map(lambda d: "AL" if d == 0 else str(d))
    m["display"] = m.name.str.title() + " (" + m.party + ", " + m.seat + ")"
    rc = pd.read_csv(RES / "issues_rollcalls_119H.csv", parse_dates=["date"])
    pa = pd.read_csv(RES / "issues_policy_areas_119H.csv")
    mv = pd.read_csv(RES / "issues_member_votes_119H.csv.gz")
    return m, rc, pa, mv


m, rc, pa, mv = load()
voting = m[~m.delegate]

st.title("How your representative votes")
st.markdown(
    "Every recorded vote in the U.S. House is public. This page summarizes the **119th Congress "
    f"(January 2025 – {rc.date.max():%B %Y})** from the official record, the same way for every member "
    "of both parties. It doesn't rate anyone as good or bad. It shows what they did, so you can decide.  \n"
    "**Election Day is Tuesday, November 3.** Find your polling place and registration status at "
    "[vote.gov](https://vote.gov).")

tab1, tab2, tab3 = st.tabs(["Look up a representative", "Issues: where the parties agree and split",
                            "About this data"])

with tab1:
    c1, c2 = st.columns([1, 2])
    state = c1.selectbox("State", sorted(m.state_name.unique()), index=None, placeholder="Choose your state")
    opts = m[m.state_name == state] if state else m
    who = c2.selectbox("Representative", opts.sort_values(["state", "district"]).display, index=None,
                       placeholder="Choose a member (or type a name)")
    st.caption("Not sure who represents you? Enter your address at "
               "[house.gov/representatives/find-your-representative](https://www.house.gov/representatives/find-your-representative).")
    if who:
        p = m[m.display == who].iloc[0]
        same = voting[voting.party == p.party]
        st.subheader(p["name"].title())
        st.write(f"{p.party_name} · {p.state_name}, district {p.seat.split('-')[1]}"
                 + (" · Delegate (votes on amendments only, not final passage)" if p.delegate else ""))
        k1, k2, k3 = st.columns(3)
        k1.metric("Votes cast", f"{int(p.votes_cast):,}")
        k2.metric("Missed votes", f"{p.missed_share:.1%}",
                  help="Share of roll calls held while they were serving where they didn't vote. "
                       f"Median {p.party_name}: {same.missed_share.median():.1%}.")
        k3.metric("Voted with the other party", f"{p.against_party_share:.1%}" if pd.notna(p.against_party_share) else "n/a",
                  help="On votes where the two parties' majorities disagreed (procedural rule votes excluded): "
                       "the share where this member sided with the other party's majority. "
                       f"Median {p.party_name}: {same.against_party_share.median():.1%}.")
        st.markdown(
            f"On the **{int(rc[(rc.D_yes > .5) & (rc.R_yes > .5)].shape[0])} votes where most Democrats and most "
            f"Republicans both voted yes**, {p['name'].title().split(',')[0]} voted yes "
            f"**{p.bipartisan_yes_share:.0%}** of the time (median {p.party_name}: {same.bipartisan_yes_share.median():.0%}).")

        st.markdown("**Where they sit among all House members** (each dot is one member; the highlighted dot is "
                    f"{p['name'].title().split(',')[0]}):")
        base = alt.Chart(voting.dropna(subset=["against_party_share"])).mark_circle(size=40, opacity=0.45).encode(
            x=alt.X("against_party_share:Q", title="Voted with the other party (share of party-split votes)",
                    axis=alt.Axis(format="%")),
            y=alt.Y("bipartisan_yes_share:Q", title="Yes on bills both parties backed", axis=alt.Axis(format="%"),
                    scale=alt.Scale(zero=False)),
            color=alt.Color("party_name:N", scale=COLOR, title=None),
            tooltip=[alt.Tooltip("display:N", title="Member"),
                     alt.Tooltip("against_party_share:Q", title="With other party", format=".1%"),
                     alt.Tooltip("bipartisan_yes_share:Q", title="Yes on bipartisan bills", format=".0%"),
                     alt.Tooltip("missed_share:Q", title="Missed", format=".1%")])
        me = alt.Chart(m[m.display == who]).mark_point(size=260, shape="diamond", filled=True, color="black").encode(
            x="against_party_share:Q", y="bipartisan_yes_share:Q")
        st.altair_chart((base + me).properties(height=420), use_container_width=True)

        # their votes, most recent first, on substantive roll calls with a bill
        st.markdown("**Recent votes on bills** (from the official record; links go to the bill on Congress.gov):")
        st.caption("Shown for context. One vote rarely tells the whole story: a member may oppose a bill "
                   "over one provision, or vote for a version they hope to change.")
        mine = mv[mv.icpsr == p.icpsr].merge(rc, on="rollnumber").sort_values("date", ascending=False)
        mine = mine.assign(**{"Their vote": mine.vote,
                              "Democrats": mine.D_yes.map("{:.0%} yes".format),
                              "Republicans": mine.R_yes.map("{:.0%} yes".format)})
        topic = st.selectbox("Filter by policy area", sorted(mine.policy_area.dropna().unique()), index=None,
                             placeholder="All policy areas")
        if topic:
            mine = mine[mine.policy_area == topic]
        st.dataframe(mine[["date", "bill_number", "bill_title", "Their vote", "Democrats", "Republicans",
                           "policy_area", "bill_url"]],
                     hide_index=True, height=420,
                     column_config={"bill_url": st.column_config.LinkColumn("Bill", display_text="Congress.gov"),
                                    "date": st.column_config.DateColumn("Date"),
                                    "bill_number": "Bill", "bill_title": "Title", "policy_area": "Policy area"})

with tab2:
    st.markdown(
        "For each policy area (assigned to every bill by the Library of Congress), how differently the two "
        "parties voted. **0 = both parties voted the same way; 1 = every Democrat on one side, every "
        "Republican on the other.** Final-passage, fast-track and amendment votes only.")
    t = pa[pa.votes >= 5].copy()
    t["label"] = t.policy_area + " (" + t.votes.astype(str) + " votes)"
    bars = alt.Chart(t).mark_bar(color="#6b6b6b").encode(
        x=alt.X("median_gap:Q", title="Typical gap between the parties (median)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=420)),
        tooltip=["policy_area", "votes", alt.Tooltip("median_gap:Q", format=".2f"),
                 alt.Tooltip("bipartisan_share:Q", title="Bipartisan votes", format=".0%"),
                 alt.Tooltip("party_line_share:Q", title="Party-line votes", format=".0%")])
    st.altair_chart(bars.properties(height=28 * len(t) + 40), use_container_width=True)
    area = st.selectbox("See the votes in one area", sorted(t.policy_area), index=None)
    if area:
        v = rc[(rc.policy_area == area) & rc.type.isin(["passage", "suspension", "amendment"])].sort_values("gap")
        v = v.assign(Democrats=v.D_yes.map("{:.0%} yes".format), Republicans=v.R_yes.map("{:.0%} yes".format))
        st.dataframe(v[["date", "bill_number", "bill_title", "vote_question", "Democrats", "Republicans", "bill_url"]],
                     hide_index=True, column_config={"bill_url": st.column_config.LinkColumn("Bill", display_text="Congress.gov"),
                                                     "date": st.column_config.DateColumn("Date")})

with tab3:
    st.markdown(f"""
**Votes:** [Voteview](https://voteview.com) (Lewis, Poole, Rosenthal, Boche, Rudkin & Sonnet), the standard
public database of congressional roll calls. **Bill topics and titles:** [Congress.gov](https://www.congress.gov)
(Library of Congress). Every vote is used, including near-unanimous ones.

- *Voted with the other party*: on votes where most Democrats and most Republicans voted opposite ways
  ({int(rc[((rc.D_yes > .5) != (rc.R_yes > .5)) & (rc.type != 'rule')].shape[0])} votes, procedural rule votes
  excluded), the share where the member sided with the other party's majority.
- *Missed votes*: roll calls held while they were serving where they didn't vote. Absences can have
  ordinary reasons (illness, family, a campaign, a vacancy before a special election).
- Delegates from DC and the territories vote only on amendments, so their numbers aren't comparable.

This page describes; it doesn't endorse. Code: [github.com/godot107/ideology-clusters](https://github.com/godot107/ideology-clusters).
Write-up: [willieman.com](https://willieman.com/blog/).
""")
