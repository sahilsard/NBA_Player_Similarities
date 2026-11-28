"""
load_bbref.py -- Fallback Phase 1 loader: Basketball-Reference CSVs -> player_seasons.csv

Use this when stats.nba.com is unreachable. You download the tables by hand
once, drop them in data/bbref/, and this script cleans and merges them into
the same player-season table the NBA-API script produces.

Expected files in data/bbref/ (.csv or .xlsx), one per table per season:

    per_poss_2023-24.csv      Per 100 Poss table   (required)
    advanced_2023-24.csv      Advanced table       (required)
    shooting_2023-24.csv      Shooting table       (optional: shot-distance mix)

...and the same for 2024-25, 2025-26, 2026-27.

Run from the repo root:   python load_bbref.py
Output:                   data/processed/player_seasons.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT_SEASON = "2026-27"
IN_DIR = Path("data/bbref")
OUT_DIR = Path("data/processed")
MIN_FLOOR_FULL_SEASON = 500
MIN_FLOOR_PARTIAL_SEASON = 150

PER_POSS_MAP = {
    "FG": "FGM_P100", "FGA": "FGA_P100", "3P": "FG3M_P100", "3PA": "FG3A_P100",
    "FT": "FTM_P100", "FTA": "FTA_P100", "ORB": "OREB_P100", "DRB": "DREB_P100",
    "TRB": "REB_P100", "AST": "AST_P100", "STL": "STL_P100", "BLK": "BLK_P100",
    "TOV": "TOV_P100", "PF": "PF_P100", "PTS": "PTS_P100",
    "3P%": "FG3_PCT", "FT%": "FT_PCT", "ORtg": "OFF_RATING", "DRtg": "DEF_RATING",
}
ADVANCED_MAP = {
    "PER": "PER", "TS%": "TS_PCT", "3PAr": "FG3A_RATE", "FTr": "FT_RATE",
    "ORB%": "OREB_PCT", "DRB%": "DREB_PCT", "TRB%": "REB_PCT", "AST%": "AST_PCT",
    "STL%": "STL_PCT", "BLK%": "BLK_PCT", "TOV%": "TM_TOV_PCT", "USG%": "USG_PCT",
    "OWS": "OWS", "DWS": "DWS", "WS": "WS", "WS/48": "WS_48",
    "OBPM": "OBPM", "DBPM": "DBPM", "BPM": "BPM", "VORP": "VORP",
}


# ----------------------------------------------------------------------------
def find_file(kind: str, season: str, required: bool = True) -> Path | None:
    """Look in data/bbref/ first, then the repo root. Tolerates stray suffixes
    (e.g. 'per_poss_2024-25.csv .xls') as long as the name starts with <kind>_<season>."""
    for folder in (IN_DIR, Path(".")):
        if folder.exists():
            hits = sorted(p for p in folder.glob(f"{kind}_{season}*") if p.is_file())
            if hits:
                return hits[0]
    if required:
        raise FileNotFoundError(
            f"Missing {kind}_{season}.csv  (looked in {IN_DIR}/ and the current folder)"
        )
    return None


def _is_html(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(4096).lower()
    return b"<table" in head or b"<html" in head


def _read_html_table(path: Path) -> pd.DataFrame:
    """Basketball-Reference's 'Excel Workbook' download is really an HTML table."""
    t = pd.read_html(path)[0]
    if isinstance(t.columns, pd.MultiIndex):  # shooting table: group row + header row
        cols = [str(c) if str(g).startswith("Unnamed") else f"{g}|{c}" for g, c in t.columns]
    else:
        cols = [str(c) for c in t.columns]
    t.columns = cols
    t = t.astype(str)
    t = t.loc[:, ~pd.Index(t.columns).duplicated(keep="first")]
    t = t[t["Player"].str.strip().ne("Player") & t["Player"].str.strip().ne("") & t["Player"].ne("nan")]
    return t.reset_index(drop=True)


def read_table(path: Path) -> pd.DataFrame:
    """Read a Basketball-Reference export (HTML, CSV or real Excel), tolerating a group-header row."""
    if _is_html(path):
        return _read_html_table(path)
    if path.suffix not in (".csv",) and not path.name.lower().endswith(".csv"):
        raw = pd.read_excel(path, header=None)
    else:  # csv module + padding: the group-header row can have a different field count
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
        width = max(len(r) for r in rows)
        raw = pd.DataFrame([r + [""] * (width - len(r)) for r in rows])
    raw = raw.fillna("").astype(str)
    hdr = next((i for i in range(min(5, len(raw))) if "Player" in set(raw.iloc[i].str.strip())), None)
    if hdr is None:
        raise ValueError(f"{path.name}: could not find a header row containing 'Player'")
    cols = raw.iloc[hdr].str.strip().tolist()
    groups = None
    if hdr > 0:  # shooting-style two-row header: group names sit one row above
        groups = pd.Series(raw.iloc[hdr - 1].str.strip().replace("", np.nan).replace("nan", np.nan)).ffill().fillna("")
        identity = {"Rk", "Player", "Age", "Team", "Tm", "Pos", "G", "GS", "MP", "Player-additional"}
        cols = [f"{g}|{c}" if g and c not in identity else c for g, c in zip(groups, cols)]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    df = df[df["Player"].str.strip().ne("Player") & df["Player"].str.strip().ne("") & df["Player"].ne("nan")]
    return df.reset_index(drop=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")


def collapse_traded(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Players traded mid-season appear as a total row (2TM/3TM/TOT) plus one row per team.
    Keep the total row's stats; set the team to the last team he played for."""
    d = df.copy()
    d["_pid"] = d["Player-additional"] if "Player-additional" in d.columns else d["Player"]
    is_total = d[team_col].str.match(r"^(\d+TM|TOT)$")
    keep = []
    for _, g in d.groupby("_pid", sort=False):
        if len(g) == 1:
            keep.append(g.iloc[0])
            continue
        tot = g[is_total.loc[g.index]]
        teams = g[~is_total.loc[g.index]]
        row = (tot.iloc[0] if len(tot) else g.iloc[0]).copy()
        if len(teams):
            row[team_col] = teams.iloc[-1][team_col]
        keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True)


def team_col_of(df: pd.DataFrame) -> str:
    for c in ("Team", "Tm"):
        if c in df.columns:
            return c
    raise ValueError("No Team/Tm column found")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    tc = team_col_of(df)
    df = collapse_traded(df, tc)
    df["PLAYER_NAME"] = df["Player"].str.replace("*", "", regex=False).str.strip()
    df["PLAYER_ID"] = df["Player-additional"] if "Player-additional" in df.columns else df["PLAYER_NAME"]
    df["TEAM_ABBREVIATION"] = df[tc]
    return df


def shooting_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """Best-effort shot-distance mix. Returns None (with a warning) if the layout isn't recognized."""
    out = pd.DataFrame({"PLAYER_ID": df["PLAYER_ID"]})
    found = 0
    for c in df.columns:
        grp, _, name = c.partition("|")
        g = grp.lower()
        if c == "Dist.":
            out["AVG_SHOT_DIST"] = num(df[c]); found += 1
        elif "ast'd" in g and name in {"2P", "3P"}:
            out[f"PCT_FG_ASSISTED_{name}"] = num(df[c]); found += 1
        elif "% of fga" in g and name in {"0-3", "3-10", "10-16", "16-3P", "3P"}:
            out[f"SHOT_FREQ_{name.replace('-', '_')}"] = num(df[c]); found += 1
        elif "fg%" in g and "distance" in g and name in {"0-3", "3-10", "10-16", "16-3P", "3P"}:
            out[f"SHOT_FG_{name.replace('-', '_')}"] = num(df[c]); found += 1
        elif "dunk" in g and name.startswith("%FGA"):
            out["DUNK_FGA_PCT"] = num(df[c]); found += 1
        elif "corner" in g and name.startswith("%3PA"):
            out["CORNER3_SHARE_OF_3PA"] = num(df[c]); found += 1
    if found == 0:
        print("  WARNING: shooting table layout not recognized; skipping shot-distance features")
        return None
    return out.drop_duplicates("PLAYER_ID")


def build_season(season: str) -> pd.DataFrame:
    pp = prep(read_table(find_file("per_poss", season)))
    adv = prep(read_table(find_file("advanced", season)))

    out = pd.DataFrame({
        "PLAYER_ID": pp["PLAYER_ID"], "PLAYER_NAME": pp["PLAYER_NAME"],
        "TEAM_ABBREVIATION": pp["TEAM_ABBREVIATION"],
        "AGE": num(pp.get("Age", pd.Series(np.nan, index=pp.index))),
        "POS": pp.get("Pos"),
        "GP": num(pp["G"]), "MIN_TOTAL": num(pp["MP"]),
    })
    for src, dst in PER_POSS_MAP.items():
        if src in pp.columns:
            out[dst] = num(pp[src]).values

    a = pd.DataFrame({"PLAYER_ID": adv["PLAYER_ID"]})
    for src, dst in ADVANCED_MAP.items():
        if src in adv.columns:
            a[dst] = num(adv[src]).values
    out = out.merge(a.drop_duplicates("PLAYER_ID"), on="PLAYER_ID", how="left")

    sp = find_file("shooting", season, required=False)
    if sp is not None:
        sh = shooting_features(prep(read_table(sp)))
        if sh is not None:
            out = out.merge(sh, on="PLAYER_ID", how="left")

    # Derived (only where the advanced table didn't already supply them)
    out["MPG"] = out["MIN_TOTAL"] / out["GP"].replace(0, np.nan)
    if "FT_RATE" not in out:
        out["FT_RATE"] = out["FTA_P100"] / out["FGA_P100"].replace(0, np.nan)
    if "FG3A_RATE" not in out:
        out["FG3A_RATE"] = out["FG3A_P100"] / out["FGA_P100"].replace(0, np.nan)

    partial = season == CURRENT_SEASON
    floor = MIN_FLOOR_PARTIAL_SEASON if partial else MIN_FLOOR_FULL_SEASON
    out.insert(0, "SEASON", season)
    out["IS_PARTIAL_SEASON"] = partial
    out["MIN_OK"] = out["MIN_TOTAL"] >= floor
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    args = ap.parse_args(argv)

    pieces = []
    for s in args.seasons:
        print(f"Season {s}")
        df = build_season(s)
        print(f"  -> {len(df)} players, {int(df['MIN_OK'].sum())} above minutes floor")
        pieces.append(df)

    allp = pd.concat(pieces, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "player_seasons.csv"
    allp.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(allp)} rows, {allp.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
