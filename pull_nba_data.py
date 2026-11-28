"""
pull_nba_data.py -- Phase 1 data pull for the NBA Player Similarity project.

Pulls league-wide player tables from stats.nba.com (via nba_api) for the last
3 completed seasons plus the current partial season, caches each raw table to
CSV, and merges everything into one player-season feature table.

Run from the repo root:

    python pull_nba_data.py                 # pull everything (cached after first run)
    python pull_nba_data.py --refresh       # ignore the cache, re-download all
    python pull_nba_data.py --seasons 2025-26 2026-27

Outputs:
    data/raw/<table>_<season>.csv           raw cached API tables
    data/processed/player_seasons.csv       one row per player per season

Design notes:
  * League-wide endpoints only (one call per table per season), NOT one call
    per player. That is ~16 calls total instead of ~2,000.
  * Counting stats come from the Per100Possessions view, which handles the
    "how do I normalize minutes" question. Rates (TS%, USG%, AST%, ...) come
    from the Advanced table.
  * The current season's cache is refetched if older than 24 hours, since the
    partial season changes daily.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT_SEASON = "2026-27"

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# Minutes floors used to flag low-sample rows (kept in the data, not dropped).
MIN_FLOOR_FULL_SEASON = 500
MIN_FLOOR_PARTIAL_SEASON = 150

PAUSE_SECONDS = 1.5        # polite delay between API calls
MAX_RETRIES = 4
TIMEOUT = 60
CACHE_MAX_AGE_HOURS_CURRENT = 24

# stats.nba.com rejects requests that don't look like a browser.
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

# Columns to keep from each table (missing ones are skipped, not fatal).
BASE_COUNTING = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                 "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PTS"]
BASE_OTHER = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
              "FG3_PCT", "FT_PCT"]

ADV_COLS = ["GP", "MIN", "OFF_RATING", "DEF_RATING", "NET_RATING",
            "AST_PCT", "AST_TO", "AST_RATIO", "OREB_PCT", "DREB_PCT",
            "REB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "USG_PCT",
            "PACE", "PIE"]

SCORING_COLS = ["PCT_FGA_2PT", "PCT_FGA_3PT", "PCT_PTS_2PT_MR", "PCT_PTS_3PT",
                "PCT_PTS_FB", "PCT_PTS_FT", "PCT_PTS_OFF_TOV", "PCT_PTS_PAINT",
                "PCT_AST_2PM", "PCT_UAST_2PM", "PCT_AST_3PM", "PCT_UAST_3PM"]

BIO_COLS = ["PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "DRAFT_YEAR",
            "DRAFT_ROUND", "DRAFT_NUMBER"]

SHOT_ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
              "Left Corner 3", "Right Corner 3", "Above the Break 3"]


# ----------------------------------------------------------------------------
# Fetching (network) -- kept separate from the pure-pandas feature building
# ----------------------------------------------------------------------------
def _endpoint_calls():
    """Map table name -> function(season) returning a nba_api endpoint object."""
    from nba_api.stats.endpoints import (
        leaguedashplayerbiostats,
        leaguedashplayershotlocations,
        leaguedashplayerstats,
    )

    common = dict(headers=HEADERS, timeout=TIMEOUT)

    def stats(measure, per_mode):
        return lambda s: leaguedashplayerstats.LeagueDashPlayerStats(
            season=s,
            season_type_all_star="Regular Season",
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure,
            **common,
        )

    return {
        # counting stats per 100 possessions
        "base": stats("Base", "Per100Possessions"),
        # rates + total minutes/games (Totals so MIN is season total)
        "adv": stats("Advanced", "Totals"),
        # shot-type and assisted/unassisted splits
        "scoring": stats("Scoring", "Totals"),
        "bio": lambda s: leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=s, season_type_all_star="Regular Season", **common
        ),
        "shots": lambda s: leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
            season=s,
            season_type_all_star="Regular Season",
            distance_range="By Zone",
            **common,
        ),
    }


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Shot-location tables can come back with two-level headers; flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(p).strip() for p in col if str(p).strip() and not str(p).startswith("Unnamed"))
            for col in df.columns
        ]
    return df


def _cache_is_fresh(path: Path, season: str, refresh: bool) -> bool:
    if refresh or not path.exists():
        return False
    if season == CURRENT_SEASON:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < CACHE_MAX_AGE_HOURS_CURRENT
    return True


def get_frame(name: str, season: str, calls: dict, refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / f"{name}_{season}.csv"
    if _cache_is_fresh(path, season, refresh):
        print(f"  cached   {name:8s} {season}")
        return pd.read_csv(path)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  fetching {name:8s} {season} (attempt {attempt})")
            df = flatten_columns(calls[name](season).get_data_frames()[0])
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            time.sleep(PAUSE_SECONDS + random.random())
            return df
        except Exception as err:  # network errors, timeouts, throttling
            last_err = err
            wait = 5 * attempt + random.random() * 3
            print(f"    failed: {type(err).__name__}: {err}. Waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {name} for {season}: {last_err}")


# ----------------------------------------------------------------------------
# Feature building (pure pandas -- no network)
# ----------------------------------------------------------------------------
def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]


def add_shot_zone_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Turn per-zone FGM/FGA into shot-frequency and FG% by zone."""
    out = pd.DataFrame({"PLAYER_ID": shots["PLAYER_ID"]})

    def col(zone: str, stat: str) -> pd.Series:
        name = f"{zone}_{stat}"
        if name in shots.columns:
            return pd.to_numeric(shots[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=shots.index)

    zones = {
        "RIM": ["Restricted Area"],
        "PAINT": ["In The Paint (Non-RA)"],
        "MID": ["Mid-Range"],
        "CORNER3": ["Left Corner 3", "Right Corner 3"],
        "ATB3": ["Above the Break 3"],
    }
    fga = {k: sum(col(z, "FGA") for z in v) for k, v in zones.items()}
    fgm = {k: sum(col(z, "FGM") for z in v) for k, v in zones.items()}
    total = sum(fga.values()).replace(0, float("nan"))

    for k in zones:
        out[f"SHOT_FREQ_{k}"] = fga[k] / total
        out[f"SHOT_FG_{k}"] = fgm[k] / fga[k].replace(0, float("nan"))
    return out


def build_season(frames: dict[str, pd.DataFrame], season: str) -> pd.DataFrame:
    """Merge the raw tables for one season into one row per player."""
    base = frames["base"]
    out = _select(base, BASE_OTHER).copy()
    for c in BASE_COUNTING:
        if c in base.columns:
            out[f"{c}_P100"] = base[c].values

    for name, cols in (("adv", ADV_COLS), ("scoring", SCORING_COLS), ("bio", BIO_COLS)):
        sub = _select(frames[name], ["PLAYER_ID"] + cols).drop_duplicates("PLAYER_ID")
        out = out.merge(sub, on="PLAYER_ID", how="left")

    if "shots" in frames:
        out = out.merge(
            add_shot_zone_features(frames["shots"]).drop_duplicates("PLAYER_ID"),
            on="PLAYER_ID", how="left",
        )

    out = out.rename(columns={"MIN": "MIN_TOTAL"})
    out["PLAYER_WEIGHT"] = pd.to_numeric(out.get("PLAYER_WEIGHT"), errors="coerce")

    # Derived features
    out["MPG"] = out["MIN_TOTAL"] / out["GP"].replace(0, float("nan"))
    out["FT_RATE"] = out["FTA_P100"] / out["FGA_P100"].replace(0, float("nan"))
    out["FG3A_RATE"] = out["FG3A_P100"] / out["FGA_P100"].replace(0, float("nan"))

    partial = season == CURRENT_SEASON
    floor = MIN_FLOOR_PARTIAL_SEASON if partial else MIN_FLOOR_FULL_SEASON
    out.insert(0, "SEASON", season)
    out["IS_PARTIAL_SEASON"] = partial
    out["MIN_OK"] = out["MIN_TOTAL"] >= floor
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--refresh", action="store_true", help="ignore cached CSVs")
    args = ap.parse_args(argv)

    calls = _endpoint_calls()
    names = list(calls)
    pieces = []
    for season in args.seasons:
        print(f"Season {season}")
        frames = {n: get_frame(n, season, calls, args.refresh) for n in names}
        df = build_season(frames, season)
        print(f"  -> {len(df)} players, {int(df['MIN_OK'].sum())} above minutes floor")
        pieces.append(df)

    allp = pd.concat(pieces, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "player_seasons.csv"
    allp.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(allp)} rows, {allp.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
