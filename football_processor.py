import argparse
import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
from zoneinfo import ZoneInfo

import requests
from decouple import config
from pydantic.dataclasses import dataclass
from slugify import slugify
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = config("LOG_LEVEL", default="INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(levelname)s - %(message)s")

SUPABASE_URL = config("SUPABASE_URL")
SUPABASE_ANON_KEY = config("SUPABASE_ANON_KEY", default=None)
AF_API_KEY = config("AF_API_KEY")
AF_BASE_URL = config("AF_BASE_URL", default="https://v3.football.api-sports.io")

DEFAULT_TZ = config("DEFAULT_TIMEZONE", default="America/Sao_Paulo")
SEASON = str(config("AF_SEASON", default="2025"))

MAX_IDS_PER_BATCH = int(config("MAX_IDS_PER_BATCH", default=20))
REQUEST_LIMIT_DAILY = int(config("REQUEST_LIMIT_DAILY", default=7500))
LIVE_UPDATE_INTERVAL_MINUTES = int(config("LIVE_UPDATE_INTERVAL_MINUTES", default=10))

ODDS_STALE_LIVE_MIN = int(config("ODDS_STALE_LIVE_MIN", default=5))
ODDS_STALE_SOON_MIN = int(config("ODDS_STALE_SOON_MIN", default=15))
ODDS_STALE_REGULAR_MIN = int(config("ODDS_STALE_REGULAR_MIN", default=60))
ODDS_SOON_HOURS = int(config("ODDS_SOON_HOURS", default=3))
ODDS_PAST_HOURS = int(config("ODDS_PAST_HOURS", default=6))
ODDS_FUTURE_HOURS = int(config("ODDS_FUTURE_HOURS", default=72))
AF_MAX_ODDS_FETCH = int(config("AF_MAX_ODDS_FETCH", default=200))
PREDICTIONS_LOOKAHEAD_HOURS = int(config("PREDICTIONS_LOOKAHEAD_HOURS", default=24 * 7))

assert SUPABASE_URL and SUPABASE_ANON_KEY
assert AF_API_KEY


# ---------------------------------------------------------------------------
# Enums & Helpers
# ---------------------------------------------------------------------------

class CommandType(Enum):
    FULL_LOAD = "full"
    INCREMENTAL = "incremental"
    LEAGUE_SYNC = "leagues"
    ODDS_SYNC = "odds"
    AUTO = "auto"


class FixtureStatus(Enum):
    NOT_STARTED = "NS"
    TO_BE_DEFINED = "TBD"
    POSTPONED = "PST"
    FIRST_HALF = "1H"
    HALFTIME = "HT"
    SECOND_HALF = "2H"
    EXTRA_TIME = "ET"
    BREAK_TIME = "BT"
    PENALTY = "P"
    SUSPENDED = "SUSP"
    INTERRUPTED = "INT"
    LIVE = "LIVE"
    FULL_TIME = "FT"
    AFTER_EXTRA_TIME = "AET"
    PENALTY_FINISHED = "PEN"
    ABANDONED = "ABD"
    CANCELLED = "CAN"
    AWARDED = "AWD"
    WALKOVER = "WO"
    CANCELLED_ALT = "CANC"

    @classmethod
    def get_finished_statuses(cls) -> Set[str]:
        return {cls.FULL_TIME.value, cls.AFTER_EXTRA_TIME.value, cls.PENALTY_FINISHED.value,
                cls.ABANDONED.value, cls.CANCELLED.value, cls.AWARDED.value,
                cls.WALKOVER.value, cls.CANCELLED_ALT.value}

    @classmethod
    def get_live_statuses(cls) -> Set[str]:
        return {cls.FIRST_HALF.value, cls.HALFTIME.value, cls.SECOND_HALF.value,
                cls.EXTRA_TIME.value, cls.BREAK_TIME.value, cls.PENALTY.value,
                cls.SUSPENDED.value, cls.INTERRUPTED.value, cls.LIVE.value}

    @classmethod
    def get_upcoming_statuses(cls) -> Set[str]:
        return {cls.TO_BE_DEFINED.value, cls.NOT_STARTED.value, cls.POSTPONED.value}


class RequestTracker:
    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self.request_count = 0
        self.start_time = datetime.now()

    def increment(self) -> bool:
        self.request_count += 1
        if self.request_count > self.daily_limit:
            logging.error(f"Daily API limit exceeded: {self.request_count}/{self.daily_limit}")
            return False
        return True

    def get_usage_percentage(self) -> float:
        return (self.request_count / self.daily_limit) * 100

    def should_warn(self) -> bool:
        return self.get_usage_percentage() > 80


REQUEST_TRACKER = RequestTracker(REQUEST_LIMIT_DAILY)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LeagueRow:
    id: int
    name: Optional[str] = None
    country: Optional[str] = None
    type: Optional[str] = None
    logo: Optional[str] = None


@dataclass
class TeamRow:
    id: int
    name: Optional[str] = None
    logo: Optional[str] = None


@dataclass
class FixtureRow:
    id: int
    league_id: Optional[int] = None
    league_name: Optional[str] = None
    season: Optional[int] = None
    round_name: Optional[str] = None
    venue_name: Optional[str] = None
    status: str = "NS"
    start_utc: Optional[str] = None
    start_local: Optional[str] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    seo_slug: str = ""
    last_update: str = ""

    def is_live(self) -> bool:
        return self.status in FixtureStatus.get_live_statuses()

    def is_finished(self) -> bool:
        return self.status in FixtureStatus.get_finished_statuses()

    def is_recent_finished(self) -> bool:
        if not self.is_finished() or not self.start_utc:
            return False
        try:
            game_time = datetime.fromisoformat(self.start_utc.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - game_time) <= timedelta(hours=6)
        except:
            return False


# ---------------------------------------------------------------------------
# API & DB Services
# ---------------------------------------------------------------------------

class APIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": api_key,
            "Accept": "application/json"
        })

    def get(self, path: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        if not REQUEST_TRACKER.increment():
            raise Exception("Daily request limit exceeded")

        url = f"{self.base_url}/{path.lstrip('/')}"
        logging.info(f"API Request #{REQUEST_TRACKER.request_count}: {path} - {params}")

        response = self.session.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        if REQUEST_TRACKER.should_warn():
            logging.warning(f"High API usage: {REQUEST_TRACKER.get_usage_percentage():.1f}%")

        time.sleep(0.12)
        return response.json()


class DatabaseService:
    def __init__(self, client: Client):
        self.client = client

    # --------------------------- League / Fixture queries ---------------------------

    def get_active_leagues(self) -> List[int]:
        try:
            seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            result = self.client.table("fixtures").select("league_id").gte(
                "start_utc_ts", seven_days_ago
            ).not_.is_("league_id", "null").execute()
            league_ids = list({row["league_id"] for row in result.data})
            logging.info(f"Found {len(league_ids)} active leagues")
            return league_ids
        except Exception as e:
            logging.error(f"Failed to get active leagues: {e}")
            return []

    def get_live_fixtures(self) -> List[int]:
        try:
            live_statuses = list(FixtureStatus.get_live_statuses())
            result = self.client.table("fixtures").select("id").in_("status", live_statuses).execute()
            fixture_ids = [row["id"] for row in result.data]
            logging.info(f"Found {len(fixture_ids)} live fixtures")
            return fixture_ids
        except Exception as e:
            logging.error(f"Failed to get live fixtures: {e}")
            return []

    def get_todays_fixtures(self) -> List[int]:
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
            result = self.client.table("fixtures").select("id").gte(
                "start_utc_ts", today
            ).lt("start_utc_ts", tomorrow).execute()
            fixture_ids = [row["id"] for row in result.data]
            logging.info(f"Found {len(fixture_ids)} fixtures for today")
            return fixture_ids
        except Exception as e:
            logging.error(f"Failed to get today's fixtures: {e}")
            return []

    def get_recent_finished_fixtures(self, hours: int = 6) -> List[int]:
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            finished_statuses = list(FixtureStatus.get_finished_statuses())
            result = (
                self.client.table("fixtures")
                .select("id")
                .in_("status", finished_statuses)
                .gte("last_update", cutoff)
                .execute()
            )
            fixture_ids = [row["id"] for row in result.data]
            logging.info(f"Found {len(fixture_ids)} recent finished fixtures")
            return fixture_ids
        except Exception as e:
            logging.error(f"Failed to get recent finished fixtures: {e}")
            return []

    def get_upcoming_fixtures(self, hours: int = 24) -> List[int]:
        try:
            now = datetime.now(timezone.utc).isoformat()
            future = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
            upcoming_statuses = list(FixtureStatus.get_upcoming_statuses())
            result = self.client.table("fixtures").select("id").in_(
                "status", upcoming_statuses
            ).gte("start_utc_ts", now).lte("start_utc_ts", future).execute()
            fixture_ids = [row["id"] for row in result.data]
            logging.info(f"Found {len(fixture_ids)} upcoming fixtures")
            return fixture_ids
        except Exception as e:
            logging.error(f"Failed to get upcoming fixtures: {e}")
            return []

    def get_started_but_not_updated_fixtures(self) -> List[int]:
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = (
                self.client.table("fixtures")
                .select("id")
                .eq("status", FixtureStatus.NOT_STARTED.value)
                .lt("start_utc_ts", now)
                .execute()
            )
            fixture_ids = [row["id"] for row in result.data]
            logging.info(f"Found {len(fixture_ids)} started-but-NS fixtures")
            return fixture_ids
        except Exception as e:
            logging.error(f"Failed to get started-but-NS fixtures: {e}")
            return []

    def get_fixture_meta(self, fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Return {fixture_id: {'status': str, 'start_utc_ts': iso}}"""
        meta: Dict[int, Dict[str, Any]] = {}
        if not fixture_ids:
            return meta
        try:
            res = self.client.table("fixtures").select("id,status,start_utc_ts").in_("id", fixture_ids).execute()
            for r in res.data:
                meta[r["id"]] = {"status": r.get("status"), "start_utc_ts": r.get("start_utc_ts")}
        except Exception as e:
            logging.warning(f"Failed to fetch fixture meta: {e}")
        return meta

    # --------------------------- Predictions presence ---------------------------

    def needs_predictions(self, fixture_ids: List[int]) -> List[int]:
        if not fixture_ids:
            return []
        try:
            result = self.client.table("fixture_predictions_af").select("fixture_id").in_(
                "fixture_id", fixture_ids
            ).execute()
            existing = {row["fixture_id"] for row in result.data}
            needed = [fid for fid in fixture_ids if fid not in existing]
            logging.info(f"Need predictions for {len(needed)} out of {len(fixture_ids)} fixtures")
            return needed
        except Exception as e:
            logging.error(f"Failed to check predictions: {e}")
            return fixture_ids

    # --------------------------- Odds smart helpers ---------------------------

    def get_odds_candidates(self,
                            past_hours: int,
                            soon_hours: int,
                            future_hours: int) -> List[int]:
        """
        Build a prioritized candidate set for odds:
          - live fixtures
          - fixtures starting within 'soon_hours'
          - fixtures within past_hours/future_hours window
        """
        try:
            now = datetime.now(timezone.utc)
            soon_limit = now + timedelta(hours=soon_hours)
            past_limit = now - timedelta(hours=past_hours)
            future_limit = now + timedelta(hours=future_hours)
            upcoming_statuses = list(FixtureStatus.get_upcoming_statuses())

            ids_live = self.get_live_fixtures()

            # Soon (starting within soon_hours)
            res_soon = self.client.table("fixtures").select("id").in_("status", upcoming_statuses) \
                .gte("start_utc_ts", now.isoformat()).lte("start_utc_ts", soon_limit.isoformat()).execute()
            ids_soon = [r["id"] for r in res_soon.data]

            # Window around now
            res_window = self.client.table("fixtures").select("id") \
                .gte("start_utc_ts", past_limit.isoformat()) \
                .lte("start_utc_ts", future_limit.isoformat()) \
                .not_.in_("status", list(FixtureStatus.get_finished_statuses())) \
                .execute()
            ids_window = [r["id"] for r in res_window.data]

            # Priority ordering
            ordered = list(dict.fromkeys([*ids_live, *ids_soon, *ids_window]))
            logging.info(
                f"Odds candidate fixtures: live={len(ids_live)} soon={len(ids_soon)} window={len(ids_window)} unique={len(ordered)}")
            return ordered
        except Exception as e:
            logging.error(f"Failed to build odds candidates: {e}")
            return []

    def get_last_odds_updates(self, fixture_ids: List[int], cutoff_hours: int = 168) -> Dict[int, Optional[datetime]]:
        """
        Return a map fixture_id -> last_update (datetime) using fixture_odds rows within last 'cutoff_hours'.
        """
        if not fixture_ids:
            return {}
        out: Dict[int, Optional[datetime]] = {}
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=cutoff_hours)).isoformat()
            # Pull recent odds for those fixtures (ordered desc so first-seen is latest)
            res = self.client.table("fixture_odds").select("fixture_id,update_time") \
                .in_("fixture_id", fixture_ids) \
                .gte("update_time", cutoff) \
                .order("update_time", desc=True).execute()
            for row in res.data:
                fid = row["fixture_id"]
                if fid not in out:
                    ts = row.get("update_time")
                    try:
                        out[fid] = datetime.fromisoformat(ts) if ts else None
                    except Exception:
                        out[fid] = None
            return out
        except Exception as e:
            logging.warning(f"Failed to fetch last odds updates: {e}")
            return out

    def upsert_models(self, table: str, rows: List[Dict[str, Any]], on_conflict: str) -> None:
        if not rows:
            return

        if table == "fixture_player_stats":
            seen = set()
            deduped = []
            for r in rows:
                key = (r.get("fixture_id"), r.get("team_id"), r.get("player_id"))
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            rows = deduped

        try:
            for i in range(0, len(rows), 800):
                batch = rows[i:i + 800]
                self.client.table(table).upsert(
                    batch,
                    on_conflict=on_conflict
                ).execute()
        except Exception as e:
            logging.error(f"Failed to upsert to {table}: {e}")
            raise


class Command(ABC):
    def __init__(self, api_client: APIClient, db_service: DatabaseService):
        self.api = api_client
        self.db = db_service

    @abstractmethod
    def execute(self) -> bool:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass


class LeagueSyncCommand(Command):
    def get_description(self) -> str:
        return "Synchronizing all active leagues (always refresh from API)"

    def execute(self) -> bool:
        try:
            logging.info("Starting full league synchronization (force refresh from API)")

            target_countries = [
                "Brazil", "England", "Spain", "France", "Italy", "Germany",
                "Portugal", "Netherlands", "Argentina", "Mexico", "Japan",
                "Turkey", "USA"
            ]

            leagues_data: List[Dict[str, Any]] = []

            for country in target_countries:
                try:
                    data = self.api.get("leagues", {"country": country, "current": "true"})
                    response = data.get("response", [])
                    if not response:
                        logging.warning(f"No active leagues returned for {country}")
                        continue

                    for item in response:
                        league = item.get("league", {})
                        country_info = item.get("country", {})
                        if not league or not league.get("id"):
                            continue

                        leagues_data.append({
                            "id": int(league["id"]),
                            "name": league.get("name"),
                            "country": country_info.get("name"),
                            "type": league.get("type"),
                            "logo": league.get("logo"),
                        })
                    time.sleep(0.25)

                    logging.info(f"Fetched {len(response)} leagues from {country}")

                except Exception as e:
                    logging.warning(f"Failed to fetch leagues for {country}: {e}")
                    continue

            if not leagues_data:
                logging.warning("No leagues were fetched from API. Aborting update.")
                return False

            unique_leagues = {l["id"]: l for l in leagues_data}.values()
            self.db.upsert_models("leagues", list(unique_leagues), "id")

            logging.info(f"Synchronized {len(unique_leagues)} active leagues successfully")
            return True

        except Exception as e:
            logging.error(f"League synchronization failed: {e}")
            return False



# ---------------------------------------------------------------------------
# Incremental updates (enhanced to fetch missing predictions and smart odds)
# ---------------------------------------------------------------------------

class IncrementalUpdateCommand(Command):
    def get_description(self) -> str:
        return "Performing incremental updates for active games"

    def execute(self) -> bool:
        try:
            logging.info("Starting incremental update")

            live_fixtures = self.db.get_live_fixtures()
            if live_fixtures:
                logging.info(f"Updating {len(live_fixtures)} live fixtures")
                self._update_fixtures_details(live_fixtures, include_full_stats=True)

            started_but_ns = self.db.get_started_but_not_updated_fixtures()
            if started_but_ns:
                logging.info(f"Updating {len(started_but_ns)} fixtures that are NS but already started")
                self._update_fixtures_details(started_but_ns, include_full_stats=True)

            recent_finished = self.db.get_recent_finished_fixtures(6)
            if recent_finished:
                logging.info(f"Updating {len(recent_finished)} recently finished fixtures")
                self._update_fixtures_details(recent_finished, include_full_stats=True)

            todays_fixtures = self.db.get_todays_fixtures()
            upcoming_window = self.db.get_upcoming_fixtures(PREDICTIONS_LOOKAHEAD_HOURS)

            # Scope for predictions/odds
            fixture_scope = list({
                *live_fixtures,
                *started_but_ns,
                *recent_finished,
                *todays_fixtures,
                *upcoming_window
            })

            # Predictions ONLY for fixtures missing in DB
            if fixture_scope:
                needed_predictions = self.db.needs_predictions(fixture_scope)
                if needed_predictions:
                    logging.info(f"Fetching predictions for {len(needed_predictions)} fixtures (missing in DB)")
                    self._fetch_predictions(needed_predictions)

            # Smart odds refresh (missing or stale)
            try:
                self._smart_sync_odds()
            except Exception as e:
                logging.warning(f"Failed to smart-sync odds during incremental: {e}")

            return True

        except Exception as e:
            logging.error(f"Incremental update failed: {e}")
            return False

    # ----------------------------- Core fixture details -----------------------------

    def _deduplicate_by_keys(self, data_list: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        if not data_list:
            return []
        seen_keys = set()
        unique_data = []
        for item in data_list:
            key_parts = [str(item.get(field, "")) for field in key_fields]
            composite_key = "_".join(key_parts)
            if composite_key not in seen_keys:
                seen_keys.add(composite_key)
                unique_data.append(item)
        duplicates_removed = len(data_list) - len(unique_data)
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicates from {len(data_list)} items")
        return unique_data

    def _update_fixtures_details(self, fixture_ids: List[int], include_full_stats: bool = False):
        if not fixture_ids:
            return
        fixtures_data, events_data, lineups_data, team_stats_data, player_stats_data = [], [], [], [], []

        for batch in self._chunked(fixture_ids, MAX_IDS_PER_BATCH):
            try:
                data = self.api.get("fixtures", {"ids": "-".join(str(x) for x in batch), "timezone": DEFAULT_TZ})
                for item in data.get("response", []):
                    fixture_row, events, lineups, team_stats, player_stats = self._parse_fixture_complete(item)
                    fixtures_data.append(fixture_row)
                    events_data.extend(events)
                    lineups_data.extend(lineups)
                    if include_full_stats:
                        team_stats_data.extend(team_stats)
                        player_stats_data.extend(player_stats)
            except Exception as e:
                logging.error(f"Failed to update fixture batch: {e}")
                continue

        if fixtures_data:
            self.db.upsert_models("fixtures", fixtures_data, "id")

        if events_data:
            logging.info(f"Processing {len(events_data)} events for upsert")
            unique_events_dict = {}
            for event in events_data:
                event_key = event.get("event_key")
                if event_key:
                    unique_events_dict[event_key] = event
            unique_events = list(unique_events_dict.values())
            duplicates_removed = len(events_data) - len(unique_events)
            if duplicates_removed > 0:
                logging.warning(f"Removed {duplicates_removed} duplicate events")
            if unique_events:
                try:
                    self.db.upsert_models("fixture_events", unique_events, "event_key")
                    logging.info(f"Successfully upserted {len(unique_events)} unique events")
                except Exception as e:
                    logging.error(f"Failed to upsert events in batch: {e}")
                    logging.warning("Skipping events due to batch conflict")
            else:
                logging.warning("No unique events to upsert")

        if lineups_data:
            try:
                self.db.upsert_models("fixture_lineups", lineups_data, "fixture_id,team_id")
            except Exception as e:
                logging.warning(f"Failed to upsert lineups: {e}")

        if team_stats_data:
            try:
                self.db.upsert_models("fixture_team_stats", team_stats_data, "fixture_id,team_id,type")
            except Exception as e:
                logging.warning(f"Failed to upsert team stats: {e}")

        if player_stats_data:
            try:
                self.db.upsert_models("fixture_player_stats", player_stats_data, "fixture_id,team_id,player_id")
            except Exception as e:
                logging.warning(f"Failed to upsert player stats: {e}")

        logging.info(f"Updated {len(fixtures_data)} fixtures with complete data")

    def _parse_fixture_complete(self, item: Dict[str, Any]) -> Tuple[
        Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        fixture_data = item.get("fixture", {})
        league_data = item.get("league", {})
        teams_data = item.get("teams", {})
        goals_data = item.get("goals", {})
        fixture_id = fixture_data.get("id")
        home_team = teams_data.get("home", {})
        away_team = teams_data.get("away", {})

        start_utc = fixture_data.get("date", "").replace("T", " ").replace("Z", " +00:00") if fixture_data.get(
            "date") else None
        start_local = self._to_local_iso(fixture_data.get("date"))
        status = fixture_data.get("status", {}).get("short", "NS")

        fixture_row = {
            "id": fixture_id,
            "league_id": league_data.get("id"),
            "league_name": league_data.get("name"),
            "season": league_data.get("season"),
            "round_name": league_data.get("round"),
            "venue_name": fixture_data.get("venue", {}).get("name"),
            "status": status,
            "start_utc": start_utc,
            "start_local": start_local,
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "home_team_name": home_team.get("name"),
            "away_team_name": away_team.get("name"),
            "home_score": goals_data.get("home"),
            "away_score": goals_data.get("away"),
            "seo_slug": self._make_slug(home_team.get("name"), away_team.get("name"), start_local, fixture_id),
            "last_update": datetime.now(timezone.utc).isoformat().replace("T", " "),
            "periods": fixture_data.get("periods", dict(first=None, second=None))
        }

        # Events
        events = []
        event_keys_in_fixture = set()
        for idx, ev in enumerate(item.get("events", [])):
            event_data = {
                "fixture_id": fixture_id,
                "time_elapsed": ev.get("time", {}).get("elapsed"),
                "team_id": ev.get("team", {}).get("id"),
                "player_id": ev.get("player", {}).get("id"),
                "assist_id": ev.get("assist", {}).get("id"),
                "type": ev.get("type"),
                "detail": ev.get("detail"),
                "comments": ev.get("comments")
            }
            key_parts = [
                str(fixture_id),
                str(event_data.get("time_elapsed") or "0"),
                str(event_data.get("team_id") or "0"),
                str(event_data.get("player_id") or "0"),
                str(event_data.get("type") or ""),
                str(event_data.get("detail") or ""),
                str(idx)
            ]
            event_key = "_".join(key_parts)
            if event_key in event_keys_in_fixture:
                event_key = f"{event_key}_{len(event_keys_in_fixture)}"
            event_keys_in_fixture.add(event_key)
            event_data["event_key"] = event_key
            events.append(event_data)

        # Lineups
        lineups = []
        for lineup in item.get("lineups", []):
            team = lineup.get("team", {})
            coach = lineup.get("coach", {})
            lineups.append({
                "fixture_id": fixture_id,
                "team_id": team.get("id"),
                "formation": lineup.get("formation"),
                "coach_id": coach.get("id"),
                "coach_name": coach.get("name")
            })

        # Team stats
        team_stats = []
        for stat_block in item.get("statistics", []):
            team = stat_block.get("team", {})
            for stat in stat_block.get("statistics", []):
                team_stats.append({
                    "fixture_id": fixture_id,
                    "team_id": team.get("id"),
                    "type": stat.get("type"),
                    "value": stat.get("value")
                })

        # Player stats
        player_stats = []
        for player_block in item.get("players", []):
            team_id = player_block.get("team", {}).get("id")
            for player_data in player_block.get("players", []):
                player = player_data.get("player", {})
                for stat in player_data.get("statistics", []):
                    games = stat.get("games", {})
                    shots = stat.get("shots", {})
                    goals = stat.get("goals", {})
                    passes = stat.get("passes", {})
                    tackles = stat.get("tackles", {})
                    duels = stat.get("duels", {})
                    dribbles = stat.get("dribbles", {})
                    fouls = stat.get("fouls", {})
                    cards = stat.get("cards", {})
                    player_stats.append({
                        "fixture_id": fixture_id,
                        "team_id": team_id,
                        "player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "minutes": games.get("minutes"),
                        "rating": games.get("rating"),
                        "shots_total": shots.get("total"),
                        "shots_on": shots.get("on"),
                        "goals": goals.get("total"),
                        "assists": goals.get("assists"),
                        "passes_total": passes.get("total"),
                        "key_passes": passes.get("key"),
                        "tackles": tackles.get("total"),
                        "duels_total": duels.get("total"),
                        "duels_won": duels.get("won"),
                        "dribbles_attempts": dribbles.get("attempts"),
                        "dribbles_success": dribbles.get("success"),
                        "fouls_drawn": fouls.get("drawn"),
                        "fouls_committed": fouls.get("committed"),
                        "cards_yellow": cards.get("yellow"),
                        "cards_red": cards.get("red")
                    })
        return fixture_row, events, lineups, team_stats, player_stats

    # ----------------------------- Predictions (robust) -----------------------------

    def _fetch_predictions(self, fixture_ids: List[int]):
        predictions = []
        for fid in fixture_ids:
            try:
                data = self.api.get("predictions", {"fixture": fid})
                for item in data.get("response", []):
                    prediction = self._parse_prediction(item, fid)
                    if prediction:
                        predictions.append(prediction)
            except Exception as e:
                logging.warning(f"Failed to fetch prediction for fixture {fid}: {e}")
                continue

        if predictions:
            self.db.upsert_models("fixture_predictions_af", predictions, "fixture_id")
            logging.info(f"Saved {len(predictions)} predictions")

    def _parse_prediction(self, item: Dict[str, Any], fixture_id: int) -> Optional[Dict[str, Any]]:
        try:
            p_home, p_draw, p_away = self._try_implied_probs_from_odds(fixture_id)
            source = "odds_implied" if p_home is not None else None

            if p_home is None:
                p_home, p_draw, p_away = self._estimate_probs_from_payload(item)
                source = "poisson_payload"

            if p_home is None:
                comp_total = item.get("comparison", {}).get("total", {})
                if comp_total:
                    th = self._pct_to_unit(comp_total.get("home")) or 0.0
                    ta = self._pct_to_unit(comp_total.get("away")) or 0.0
                    d = max(0.18, min(0.35, 1.0 - th - ta))
                    rem = max(1e-9, 1.0 - d)
                    total_ha = max(1e-9, th + ta)
                    p_home, p_away, p_draw = (th / total_ha) * rem, (ta / total_ha) * rem, d
                    source = "api_comparison"

            total = (p_home or 0) + (p_draw or 0) + (p_away or 0)
            if total <= 0:
                p_home = p_draw = p_away = 1 / 3
                source = "fallback_default"
            else:
                p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

            predictions_data = item.get("predictions", {})
            winner = predictions_data.get("winner", {})
            advice = predictions_data.get("advice")

            return {
                "fixture_id": fixture_id,
                "home_win_prob": round(float(p_home), 6),
                "draw_prob": round(float(p_draw), 6),
                "away_win_prob": round(float(p_away), 6),
                "advice": advice or source,
                "winner_id": winner.get("id"),
                "winner_name": winner.get("name"),
                "winner_comment": winner.get("comment"),
                "source": source,
                "raw": item,
                "last_update": datetime.now(timezone.utc).isoformat().replace("T", " ")
            }

        except Exception as e:
            logging.error(f"Failed to parse prediction: {e}")
            return None

    def _try_implied_probs_from_odds(self, fixture_id: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            try:
                best = self.db.client.table("vw_odds_1x2_best").select("*").eq("fixture_id", fixture_id).limit(
                    1).execute()
                if best.data:
                    b = best.data[0]
                    home = float(b.get("home_best_odd")) if b.get("home_best_odd") else None
                    draw = float(b.get("draw_best_odd")) if b.get("draw_best_odd") else None
                    away = float(b.get("away_best_odd")) if b.get("away_best_odd") else None
                else:
                    res = self.db.client.table("fixture_odds").select("selection_key,odd") \
                        .eq("fixture_id", fixture_id).eq("market_code", "1X2").execute()
                    home = min((float(r["odd"]) for r in res.data if r["selection_key"] == "HOME"), default=None)
                    draw = min((float(r["odd"]) for r in res.data if r["selection_key"] == "DRAW"), default=None)
                    away = min((float(r["odd"]) for r in res.data if r["selection_key"] == "AWAY"), default=None)
            except Exception:
                return (None, None, None)

            if not (home and draw and away):
                return (None, None, None)

            invs = [1.0 / home, 1.0 / draw, 1.0 / away]
            s = sum(invs)
            if s <= 0:
                return (None, None, None)
            p = [x / s for x in invs]
            return (p[0], p[1], p[2])
        except Exception:
            return (None, None, None)

    def _estimate_probs_from_payload(self, item: Dict[str, Any]) -> Tuple[
        Optional[float], Optional[float], Optional[float]]:
        try:
            teams = item.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            comp = item.get("comparison", {})
            comp_total = comp.get("total", {})

            def parse_form_pct(s: Optional[str]) -> float:
                if not s:
                    return 0.5
                s = s.strip().replace("%", "")
                try:
                    v = float(s) / 100.0
                    return max(0.0, min(1.0, v))
                except:
                    return 0.5

            h_last5_att = parse_form_pct(home.get("last_5", {}).get("att"))
            h_last5_def = parse_form_pct(home.get("last_5", {}).get("def"))
            a_last5_att = parse_form_pct(away.get("last_5", {}).get("att"))
            a_last5_def = parse_form_pct(away.get("last_5", {}).get("def"))

            h_avg_for_home = float(
                home.get("league", {}).get("goals", {}).get("for", {}).get("average", {}).get("home", "1.1"))
            a_avg_for_away = float(
                away.get("league", {}).get("goals", {}).get("for", {}).get("average", {}).get("away", "1.1"))
            base_ha = 1.08
            ha = base_ha * (1.0 + 0.10 * (h_last5_att - a_last5_def))

            lam_home = max(0.05, min(3.5, h_avg_for_home * (1 + 0.6 * (h_last5_att - 0.5)) * (
                    1 + 0.4 * (1 - a_last5_def)) * ha))
            lam_away = max(0.05, min(3.5, a_avg_for_away * (1 + 0.6 * (a_last5_att - 0.5)) * (
                    1 + 0.4 * (1 - h_last5_def)) / ha ** 0.35))

            ct_h = self._pct_to_unit(comp_total.get("home")) if comp_total else None
            ct_a = self._pct_to_unit(comp_total.get("away")) if comp_total else None
            if ct_h is not None and ct_a is not None and (ct_h + ct_a) > 0:
                ratio = ct_h / max(1e-9, ct_a)
                curr_ratio = lam_home / max(1e-9, lam_away)
                lam_home *= (ratio / curr_ratio) ** 0.25
                lam_away *= (curr_ratio / ratio) ** 0.05
                lam_home = max(0.05, min(3.5, lam_home))
                lam_away = max(0.05, min(3.5, lam_away))

            max_goals = 10
            p_home = p_draw = p_away = 0.0
            fact = [1.0]
            for i in range(1, max_goals + 1): fact.append(fact[-1] * i)

            def pois(k, lam):
                return math.exp(-lam) * (lam ** k) / fact[k]

            for gh in range(0, max_goals + 1):
                ph = pois(gh, lam_home)
                for ga in range(0, max_goals + 1):
                    pa = pois(ga, lam_away)
                    if gh > ga:
                        p_home += ph * pa
                    elif gh == ga:
                        p_draw += ph * pa
                    else:
                        p_away += ph * pa

            s = p_home + p_draw + p_away
            if s <= 0:
                return (None, None, None)
            return (p_home / s, p_draw / s, p_away / s)
        except Exception:
            return (None, None, None)

    def _fetch_odds_with_pagination(self, fixture_id: int, bookmaker_ids: List[int]) -> List[Dict[str, Any]]:
        market_map = {
            "Match Winner": "1X2",
            "1X2": "1X2",
            "Over/Under": "OU",
            "Both Teams To Score": "BTTS",
            "Double Chance": "DC",
            "Correct Score": "CS"
        }

        all_odds = []
        page = 1
        while True:
            params = {"fixture": fixture_id, "page": page}
            if bookmaker_ids:
                params["bookmaker"] = ",".join(map(str, bookmaker_ids))
            data = self.api.get("odds", params)
            response = data.get("response", [])
            if not response:
                break

            for item in response:
                league = item.get("league", {})
                fixture = item.get("fixture", {})
                update_time = item.get("update")
                for bookmaker in item.get("bookmakers", []):
                    for market in bookmaker.get("bets", []):
                        market_name = market.get("name")
                        mapped_code = market_map.get(market_name)
                        if not mapped_code:
                            continue
                        for value in market.get("values", []):
                            all_odds.append({
                                "fixture_id": fixture_id,
                                "league_id": league.get("id"),
                                "season": league.get("season"),
                                "bookmaker_id": bookmaker.get("id"),
                                "bookmaker_name": bookmaker.get("name"),
                                "market_code": mapped_code,
                                "market_name": market_name,
                                "selection_key": value.get("value"),
                                "selection_name": value.get("value"),
                                "line": value.get("handicap"),
                                "odd": value.get("odd"),
                                "update_time": update_time,
                                "created_at": datetime.now(timezone.utc).isoformat()
                            })

            if len(response) < 10:
                break
            page += 1
        return all_odds

    def _fetch_live_odds(self, fixture_id: int, bookmaker_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
        market_map = {
            "Match Winner": "1X2",
            "1X2": "1X2",
            "Over/Under": "OU",
            "Both Teams To Score": "BTTS",
            "Double Chance": "DC",
            "Correct Score": "CS"
        }
        all_odds: List[Dict[str, Any]] = []
        page = 1
        while True:
            params: Dict[str, Any] = {"fixture": fixture_id, "page": page}
            if bookmaker_ids:
                params["bookmaker"] = ",".join(map(str, bookmaker_ids))
            data = self.api.get("odds/live", params)
            response = data.get("response", [])
            if not response:
                break
            for item in response:
                league = item.get("league", {})
                fixture = item.get("fixture", {})
                update_time = item.get("update")
                for bookmaker in item.get("bookmakers", []):
                    for market in bookmaker.get("bets", []):
                        market_name = market.get("name")
                        mapped_code = market_map.get(market_name)
                        if not mapped_code:
                            continue
                        for value in market.get("values", []):
                            all_odds.append({
                                "fixture_id": fixture.get("id"),
                                "league_id": league.get("id"),
                                "season": league.get("season"),
                                "bookmaker_id": bookmaker.get("id"),
                                "bookmaker_name": bookmaker.get("name"),
                                "market_code": mapped_code,
                                "market_name": market_name,
                                "selection_key": value.get("value"),
                                "selection_name": value.get("value"),
                                "line": value.get("handicap"),
                                "odd": value.get("odd"),
                                "update_time": update_time,
                                "created_at": datetime.now(timezone.utc).isoformat()
                            })
            if len(response) < 10:
                break
            page += 1
        return all_odds

    def _smart_sync_odds(self) -> None:
        cmd = OddsSyncCommand(self.api, self.db)

        candidates = self.db.get_odds_candidates(
            past_hours=24 * 14, soon_hours=24 * 14, future_hours=24 * 7
        )
        if not candidates:
            logging.info("No odds candidates to process")
            return

        # View com as últimas odds (se houver)
        res = self.db.client.table("v_fixture_odds_card").select(
            "fixture_id,status,last_odds_update"
        ).in_("fixture_id", candidates).execute()
        odds_meta = {row["fixture_id"]: row for row in res.data}

        # Tentativas anteriores
        attempts_res = self.db.client.table("fixture_odds_attempts").select(
            "fixture_id,last_try"
        ).in_("fixture_id", candidates).execute()
        attempts_meta = {row["fixture_id"]: row["last_try"] for row in attempts_res.data}

        now = datetime.now(timezone.utc)
        bookmaker_ids = cmd._parse_int_list(config("AF_BOOKMAKER_IDS", default=""))
        all_odds: List[Dict[str, Any]] = []
        need_fetch: List[int] = []

        for fid in candidates:
            status = odds_meta.get(fid, {}).get("status")
            if status in FixtureStatus.get_finished_statuses():
                continue

            # última atualização real das odds
            lu_str = odds_meta.get(fid, {}).get("last_odds_update")
            lu = datetime.fromisoformat(lu_str.replace("Z", "+00:00")) if lu_str else None

            # última tentativa
            lt_str = attempts_meta.get(fid)
            lt = datetime.fromisoformat(lt_str.replace("Z", "+00:00")) if lt_str else None

            # decisão
            if status in FixtureStatus.get_live_statuses():
                threshold = timedelta(minutes=10)
            else:
                threshold = timedelta(hours=3)

            ref_time = lu or lt
            if ref_time is None or (now - ref_time) >= threshold:
                need_fetch.append(fid)

        if not need_fetch:
            logging.info("All candidate odds are fresh enough")
            return

        logging.info(f"Fetching odds for {len(need_fetch)} fixtures (smart sync)")

        for fid in need_fetch:
            try:
                odds = []
                if odds_meta.get(fid, {}).get("status") in FixtureStatus.get_live_statuses():
                    odds = self._fetch_live_odds(fid, bookmaker_ids)
                else:
                    odds = self._fetch_odds_with_pagination(fid, bookmaker_ids)

                if odds:
                    all_odds.extend(odds)

                # sempre registra a tentativa
                self.db.client.table("fixture_odds_attempts").upsert({
                    "fixture_id": fid,
                    "last_try": now.isoformat()
                }).execute()

                time.sleep(0.25)

            except Exception as e:
                logging.warning(f"Failed to fetch odds for fixture {fid}: {e}")
                continue

            if len(all_odds) >= AF_MAX_ODDS_FETCH:
                break

        if all_odds:
            cmd._upsert_odds(all_odds)
            logging.info(f"Upserted {len(all_odds)} odds records")

    # ----------------------------- Utils -----------------------------

    def _pct_to_unit(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            if isinstance(value, str) and value.endswith("%"):
                return float(value[:-1]) / 100.0
            return float(value)
        except:
            return None

    def _to_local_iso(self, iso_utc: Optional[str]) -> Optional[str]:
        if not iso_utc:
            return None
        try:
            dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
            return dt.astimezone(ZoneInfo(DEFAULT_TZ)).isoformat().replace("T", " ")
        except:
            return None

    def _make_slug(self, home: Optional[str], away: Optional[str], dt_local_iso: Optional[str],
                   fixture_id: Optional[int]) -> str:
        parts = [(home or "").strip(), "x", (away or "").strip(), "palpites"]
        if dt_local_iso:
            parts.append(dt_local_iso.split(" ", 1)[0])
        if fixture_id:
            parts.append(str(fixture_id))
        return slugify(" ".join([p for p in parts if p]), lowercase=True)

    def _chunked(self, lst: List[Any], n: int) -> Iterable[List[Any]]:
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


# ---------------------------------------------------------------------------
# Full load
# ---------------------------------------------------------------------------

class FullLoadCommand(Command):
    def get_description(self) -> str:
        return "Performing full data load"

    def execute(self) -> bool:
        try:
            logging.info("Starting full load")

            league_sync = LeagueSyncCommand(self.api, self.db)
            if not league_sync.execute():
                logging.error("League sync failed, continuing anyway")

            fixtures_loaded = self._load_fixtures_by_date_range(-3, 7)
            if not fixtures_loaded:
                logging.warning("No fixtures loaded")
                return False

            self._update_priority_fixtures()
            self._update_standings()
            self._update_additional_data()

            logging.info("Full load completed")
            return True
        except Exception as e:
            logging.error(f"Full load failed: {e}")
            return False

    def _load_fixtures_by_date_range(self, start_days: int, end_days: int) -> bool:
        tz = ZoneInfo(DEFAULT_TZ)
        today = datetime.now(tz).date()
        active_leagues = self.db.get_active_leagues()
        logging.info(f"Loading fixtures for {len(active_leagues)} leagues from {start_days} to +{end_days} days")

        all_fixtures, all_teams = [], set()
        fixtures_count = 0

        for days_offset in range(start_days, end_days + 1):
            date = (today + timedelta(days=days_offset)).isoformat()
            for league_id in active_leagues:
                try:
                    data = self.api.get("fixtures", {
                        "date": date, "league": league_id, "season": SEASON, "timezone": DEFAULT_TZ
                    })
                    daily_fixtures = []
                    for item in data.get("response", []):
                        fixture = self._parse_basic_fixture(item)
                        if fixture:
                            daily_fixtures.append(fixture)
                            all_fixtures.append(fixture)
                            if fixture["home_team_id"]:
                                all_teams.add((fixture["home_team_id"], fixture["home_team_name"]))
                            if fixture["away_team_id"]:
                                all_teams.add((fixture["away_team_id"], fixture["away_team_name"]))
                    if daily_fixtures:
                        fixtures_count += len(daily_fixtures)
                        logging.info(f"Loaded {len(daily_fixtures)} fixtures for league {league_id} on {date}")
                except Exception as e:
                    logging.warning(f"Failed to load fixtures for league {league_id}, date {date}: {e}")
                    continue

        if all_fixtures:
            self.db.upsert_models("fixtures", all_fixtures, "id")
            teams_data = [{"id": tid, "name": tname} for tid, tname in all_teams if tid]
            if teams_data:
                self.db.upsert_models("teams", teams_data, "id")

            try:
                fixture_ids = [fx.get("id") for fx in all_fixtures if fx.get("id")]
                if fixture_ids:
                    missing_predictions = self.db.needs_predictions(fixture_ids)
                    if missing_predictions:
                        logging.info(
                            f"Fetching predictions for {len(missing_predictions)} fixtures loaded via full load"
                        )
                        incremental = IncrementalUpdateCommand(self.api, self.db)
                        incremental._fetch_predictions(missing_predictions)
            except Exception as e:
                logging.warning(f"Failed to backfill predictions after full load: {e}")

            logging.info(f"Full load completed: {fixtures_count} fixtures, {len(teams_data)} teams")
            return True
        return False

    def _parse_basic_fixture(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            fixture_data = item.get("fixture", {})
            league_data = item.get("league", {})
            teams_data = item.get("teams", {})
            goals_data = item.get("goals", {})
            fixture_id = fixture_data.get("id")
            if not fixture_id:
                return None
            home_team = teams_data.get("home", {})
            away_team = teams_data.get("away", {})
            start_utc = fixture_data.get("date", "").replace("T", " ").replace("Z", " +00:00") if fixture_data.get(
                "date") else None
            start_local = self._to_local_iso(fixture_data.get("date"))
            status = fixture_data.get("status", {}).get("short", "NS")
            return {
                "id": fixture_id,
                "league_id": league_data.get("id"),
                "league_name": league_data.get("name"),
                "season": league_data.get("season"),
                "round_name": league_data.get("round"),
                "venue_name": fixture_data.get("venue", {}).get("name"),
                "status": status,
                "start_utc": start_utc,
                "start_local": start_local,
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "home_team_name": home_team.get("name"),
                "away_team_name": away_team.get("name"),
                "home_score": goals_data.get("home"),
                "away_score": goals_data.get("away"),
                "seo_slug": self._make_slug(home_team.get("name"), away_team.get("name"), start_local, fixture_id),
                "last_update": datetime.now(timezone.utc).isoformat().replace("T", " ")
            }
        except Exception as e:
            logging.error(f"Failed to parse fixture: {e}")
            return None

    def _update_priority_fixtures(self):
        todays_fixtures = self.db.get_todays_fixtures()
        live_fixtures = self.db.get_live_fixtures()
        recent_finished = self.db.get_recent_finished_fixtures(12)
        priority_fixtures = list(set(todays_fixtures + live_fixtures + recent_finished))
        if priority_fixtures:
            logging.info(f"Updating {len(priority_fixtures)} priority fixtures with detailed data")
            incremental = IncrementalUpdateCommand(self.api, self.db)
            incremental._update_fixtures_details(priority_fixtures, include_full_stats=True)

        upcoming_fixtures = self.db.get_upcoming_fixtures(PREDICTIONS_LOOKAHEAD_HOURS)
        if upcoming_fixtures:
            incremental = IncrementalUpdateCommand(self.api, self.db)
            needed_predictions = self.db.needs_predictions(upcoming_fixtures)
            if needed_predictions:
                logging.info(f"Fetching predictions for {len(needed_predictions)} upcoming fixtures")
                incremental._fetch_predictions(needed_predictions)
            # Smart odds for these as well
            incremental._smart_sync_odds()

    def _update_standings(self):
        active_leagues = self.db.get_active_leagues()
        if not active_leagues:
            return
        logging.info(f"Updating standings for {len(active_leagues)} leagues")
        for league_id in active_leagues:
            try:
                data = self.api.get("standings", {"league": league_id, "season": SEASON})
                standings_data = []
                for response in data.get("response", []):
                    league = response.get("league", {})
                    for standings_group in league.get("standings", []):
                        for team_standing in standings_group:
                            team = team_standing.get("team", {})
                            all_stats = team_standing.get("all", {})
                            goals = all_stats.get("goals", {})
                            standings_data.append({
                                "league_id": league.get("id"),
                                "season": str(league.get("season", SEASON)),
                                "team_id": team.get("id"),
                                "rank": team_standing.get("rank"),
                                "points": team_standing.get("points"),
                                "played": all_stats.get("played"),
                                "win": all_stats.get("win"),
                                "draw": all_stats.get("draw"),
                                "lose": all_stats.get("lose"),
                                "goals_for": goals.get("for"),
                                "goals_against": goals.get("against"),
                                "goals_diff": team_standing.get("goalsDiff"),
                                "form": team_standing.get("form"),
                                "group_name": team_standing.get("group"),
                                "update_time": team_standing.get("update")
                            })
                if standings_data:
                    self.db.upsert_models("standings", standings_data, "league_id,season,team_id")
                    logging.info(f"Updated standings for league {league_id}: {len(standings_data)} teams")
            except Exception as e:
                logging.warning(f"Failed to update standings for league {league_id}: {e}")

    def _update_additional_data(self):
        upcoming_fixtures = self.db.get_upcoming_fixtures(7 * 24)
        if not upcoming_fixtures:
            return
        try:
            result = self.db.client.table("fixtures").select(
                "id,home_team_id,away_team_id,league_id"
            ).in_("id", upcoming_fixtures).execute()

            h2h_pairs, team_league_pairs = set(), set()
            for fixture in result.data:
                home_id = fixture.get("home_team_id")
                away_id = fixture.get("away_team_id")
                league_id = fixture.get("league_id")
                if home_id and away_id:
                    pair = tuple(sorted([home_id, away_id]))
                    h2h_pairs.add(pair)
                    if league_id:
                        team_league_pairs.add((league_id, home_id))
                        team_league_pairs.add((league_id, away_id))

            if h2h_pairs:
                self._fetch_h2h_data(list(h2h_pairs)[:50])
            if team_league_pairs:
                self._fetch_team_statistics(list(team_league_pairs)[:30])

        except Exception as e:
            logging.error(f"Failed to update additional data: {e}")

    def _fetch_h2h_data(self, h2h_pairs: List[Tuple[int, int]]):
        h2h_data = []
        for home_id, away_id in h2h_pairs:
            try:
                data = self.api.get("fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": 5})
                for fixture in data.get("response", []):
                    fixture_info = fixture.get("fixture", {})
                    league_info = fixture.get("league", {})
                    teams_info = fixture.get("teams", {})
                    goals_info = fixture.get("goals", {})
                    h2h_data.append({
                        "fixture_id": fixture_info.get("id"),
                        "league_id": league_info.get("id"),
                        "season": league_info.get("season"),
                        "date_utc": fixture_info.get("date", "").replace("T", " ").replace("Z", " +00:00"),
                        "home_team_id": teams_info.get("home", {}).get("id"),
                        "away_team_id": teams_info.get("away", {}).get("id"),
                        "home_goals": goals_info.get("home"),
                        "away_goals": goals_info.get("away"),
                        "status": fixture_info.get("status", {}).get("short")
                    })
            except Exception as e:
                logging.warning(f"Failed to fetch H2H for {home_id}-{away_id}: {e}")
                continue
        if h2h_data:
            self.db.upsert_models("h2h_fixtures", h2h_data, "fixture_id")
            logging.info(f"Updated H2H data: {len(h2h_data)} records")

    def _fetch_team_statistics(self, team_league_pairs: List[Tuple[int, int]]):
        team_stats = []
        for league_id, team_id in team_league_pairs:
            try:
                data = self.api.get("teams/statistics", {"league": league_id, "team": team_id, "season": SEASON})
                response = data.get("response", {})
                if response:
                    fixtures = response.get("fixtures", {})
                    goals = response.get("goals", {})
                    team_stats.append({
                        "league_id": league_id,
                        "team_id": team_id,
                        "season": SEASON,
                        "form": response.get("form"),
                        "fixtures_played_total": fixtures.get("played", {}).get("total"),
                        "goals_for_avg_home": goals.get("for", {}).get("average", {}).get("home"),
                        "goals_for_avg_away": goals.get("for", {}).get("average", {}).get("away"),
                        "goals_against_avg_home": goals.get("against", {}).get("average", {}).get("home"),
                        "goals_against_avg_away": goals.get("against", {}).get("average", {}).get("away"),
                        "goals_for_minute": goals.get("for", {}).get("minute", {}),
                        "goals_against_minute": goals.get("against", {}).get("minute", {}),
                        "biggest_wins": response.get("biggest", {}).get("wins", {}),
                        "clean_sheet": response.get("clean_sheet", {}),
                        "failed_to_score": response.get("failed_to_score", {}),
                        "penalty": response.get("penalty", {}),
                        "lineups": response.get("lineups", []),
                        "cards": response.get("cards", {}),
                        "updated_at": datetime.now(timezone.utc).isoformat().replace("T", " ")
                    })
            except Exception as e:
                logging.warning(f"Failed to fetch team stats for L{league_id} T{team_id}: {e}")
                continue
        if team_stats:
            self.db.upsert_models("team_statistics", team_stats, "league_id,team_id,season")
            logging.info(f"Updated team statistics: {len(team_stats)} records")

    def _to_local_iso(self, iso_utc: Optional[str]) -> Optional[str]:
        if not iso_utc:
            return None
        try:
            dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
            return dt.astimezone(ZoneInfo(DEFAULT_TZ)).isoformat().replace("T", " ")
        except:
            return None

    def _make_slug(self, home: Optional[str], away: Optional[str], dt_local_iso: Optional[str],
                   fixture_id: Optional[int]) -> str:
        parts = [(home or "").strip(), "x", (away or "").strip(), "palpites"]
        if dt_local_iso:
            parts.append(dt_local_iso.split(" ", 1)[0])
        if fixture_id:
            parts.append(str(fixture_id))
        return slugify(" ".join([p for p in parts if p]), lowercase=True)


# ---------------------------------------------------------------------------
# Odds Sync (maps only to allowed markets in schema) + SMART CANDIDATES
# ---------------------------------------------------------------------------

class OddsSyncCommand(Command):
    def get_description(self) -> str:
        return "Synchronizing odds data for open fixtures"

    def execute(self) -> bool:
        try:
            logging.info("Starting odds synchronization")

            # Smart candidate fixtures (live/soon/window)
            candidate_ids = self.db.get_odds_candidates(
                past_hours=ODDS_PAST_HOURS, soon_hours=ODDS_SOON_HOURS, future_hours=ODDS_FUTURE_HOURS
            )
            if not candidate_ids:
                logging.info("No candidate fixtures for odds sync")
                return True

            # Determine missing/stale using thresholds
            last_updates = self.db.get_last_odds_updates(candidate_ids, cutoff_hours=7 * 24)
            meta = self.db.get_fixture_meta(candidate_ids)
            now = datetime.now(timezone.utc)

            need_fetch: List[int] = []
            for fid in candidate_ids:
                if len(need_fetch) >= AF_MAX_ODDS_FETCH:
                    break
                info = meta.get(fid, {})
                status = info.get("status")
                start_ts_raw = info.get("start_utc_ts")
                try:
                    start_ts = datetime.fromisoformat(start_ts_raw) if start_ts_raw else None
                except Exception:
                    start_ts = None

                lu = last_updates.get(fid)

                if status in FixtureStatus.get_live_statuses():
                    threshold = timedelta(minutes=ODDS_STALE_LIVE_MIN)
                elif start_ts and start_ts <= now + timedelta(hours=ODDS_SOON_HOURS):
                    threshold = timedelta(minutes=ODDS_STALE_SOON_MIN)
                else:
                    threshold = timedelta(minutes=ODDS_STALE_REGULAR_MIN)

                if lu is None or (now - lu) >= threshold:
                    need_fetch.append(fid)

            if not need_fetch:
                logging.info("All candidate odds are fresh enough")
                return True

            logging.info(f"Fetching odds for {len(need_fetch)} fixtures (missing/stale)")

            all_odds = []
            bookmaker_ids = self._parse_int_list(config("AF_BOOKMAKER_IDS", default=""))
            for fixture_id in need_fetch:
                try:
                    odds = self._fetch_odds_for_fixture(fixture_id, bookmaker_ids)
                    if odds:
                        all_odds.extend(odds)
                    time.sleep(0.25)  # Rate limiting
                except Exception as e:
                    logging.warning(f"Failed to fetch odds for fixture {fixture_id}: {e}")
                    continue

            if all_odds:
                self._upsert_odds(all_odds)
                logging.info(f"Synchronized {len(all_odds)} odds records")
            else:
                logging.info("No odds data found")

            return True

        except Exception as e:
            logging.error(f"Odds sync failed: {e}")
            return False

    # --- Low-level odds fetch/flatten ---

    def _fetch_odds_for_fixture(self, fixture_id: int, bookmaker_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
        params = {"fixture": fixture_id}
        if bookmaker_ids:
            params["bookmaker"] = ",".join(str(x) for x in bookmaker_ids)
        try:
            data = self.api.get("odds", params)
            return self._flatten_odds(data)
        except Exception as e:
            logging.warning(f"Failed to fetch odds for fixture {fixture_id}: {e}")
            return []

    def _flatten_odds(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for item in payload.get("response", []):
            league = item.get("league", {})
            fixture = item.get("fixture", {})
            league_id = league.get("id")
            season = league.get("season")
            fixture_id = fixture.get("id")

            update_str = item.get("update")
            try:
                update_time = datetime.fromisoformat(
                    update_str.replace("Z", "+00:00")).isoformat() if update_str else None
            except:
                update_time = None

            for bookmaker in item.get("bookmakers", []):
                bk_id = bookmaker.get("id")
                bk_name = (bookmaker.get("name") or "").strip() or str(bk_id)

                for bet in bookmaker.get("bets", []):
                    bet_id = bet.get("id")
                    bet_name = (bet.get("name") or "").strip()
                    market_code = self._get_market_code(bet_id, bet_name)
                    if not market_code:
                        continue

                    for value in bet.get("values", []):
                        v_raw = value.get("value")
                        v_str = str(v_raw).strip() if v_raw is not None else ""
                        odd = self._to_decimal(value.get("odd"))
                        if odd is None:
                            continue

                        selection_key, line_value = self._parse_selection(market_code, v_str)
                        if not selection_key:
                            continue

                        line_num = float(line_value) if line_value is not None else None
                        out.append({
                            "fixture_id": fixture_id,
                            "league_id": league_id,
                            "season": season,
                            "bookmaker_id": bk_id,
                            "bookmaker_name": bk_name,
                            "market_code": market_code,
                            "market_name": bet_name or None,
                            "selection_key": selection_key,
                            "selection_name": v_str or None,
                            "line": line_num,
                            "line_key": line_num if line_num is not None else -1.0,
                            "odd": float(odd),
                            "update_time": update_time,
                            "created_at": datetime.now(timezone.utc).isoformat()
                        })
        return out

    def _get_market_code(self, bet_id: int, bet_name: str) -> Optional[str]:
        name = (bet_name or "").strip().lower()
        if bet_id == 1 or "match winner" in name:
            return "1X2"
        elif bet_id == 8 or "both teams score" in name:
            return "BTTS"
        elif bet_id == 5 and "over/under" in name:
            return "OU"
        elif bet_id == 12 or "double chance" in name:
            return "DC"
        elif bet_id == 10 or "exact score" in name:
            return "CS"
        return None

    def _parse_selection(self, market_code: str, value_str: str) -> Tuple[Optional[str], Optional[Decimal]]:
        v = value_str.strip().lower()
        if market_code == "1X2":
            if v == "home": return "HOME", None
            if v == "draw": return "DRAW", None
            if v == "away": return "AWAY", None
        elif market_code == "BTTS":
            if v == "yes": return "YES", None
            if v == "no": return "NO", None
        elif market_code == "OU":
            m = re.match(r"^(over|under)\s+([0-9]+(?:\.[0-9]+)?)$", v)
            if m:
                side = m.group(1).upper()
                line = self._to_decimal(m.group(2))
                return side, line
        elif market_code == "DC":
            if v == "home/draw": return "1X", None
            if v == "home/away": return "12", None
            if v == "draw/away": return "X2", None
        elif market_code == "CS":
            if re.match(r"^\d+:\d+$", value_str): return value_str, None
        return None, None

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float, Decimal)):
                return Decimal(str(value))
            elif isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
                return Decimal(value)
        except (InvalidOperation, ValueError):
            return None
        return None

    def _parse_int_list(self, env_value: str) -> Optional[List[int]]:
        if not env_value:
            return None
        ids = []
        for chunk in env_value.split(","):
            chunk = chunk.strip()
            if chunk.isdigit():
                ids.append(int(chunk))
        return ids if ids else None

    def _dedupe_odds(self, odds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for o in odds:
            key = (
                o.get("fixture_id"),
                o.get("market_code"),
                o.get("bookmaker_id"),
                o.get("selection_key"),
                o.get("line")
            )
            if key not in seen:
                seen.add(key)
                deduped.append(o)
        return deduped

    def _upsert_odds(self, odds: List[Dict[str, Any]]) -> None:
        if not odds:
            return
        clean_odds = self._dedupe_odds(odds)
        payload = []
        for o in clean_odds:
            r = dict(o)
            r.pop("line_key", None)
            payload.append(r)
        try:
            for i in range(0, len(payload), 800):
                batch = payload[i:i + 800]
                self.db.client.table("fixture_odds").upsert(
                    batch,
                    on_conflict="fixture_id,market_code,bookmaker_id,line_key,selection_key"
                ).execute()
            logging.info(f"Upserted {len(payload)} odds records")
        except Exception as e:
            logging.error(f"Failed to upsert odds: {e}")
            raise


# ---------------------------------------------------------------------------
# Auto command
# ---------------------------------------------------------------------------

class AutoCommand(Command):
    def get_description(self) -> str:
        return "Auto-detecting optimal update strategy"

    def execute(self) -> bool:
        try:
            now = datetime.now()
            hour = now.hour

            if hour in [2, 3]:
                logging.info("Auto mode: Running full load (early morning)")
                command = FullLoadCommand(self.api, self.db)
            elif self._has_live_games():
                logging.info("Auto mode: Running incremental update (live games detected)")
                command = IncrementalUpdateCommand(self.api, self.db)
            elif 6 <= hour <= 23:
                logging.info("Auto mode: Running incremental update (normal hours)")
                command = IncrementalUpdateCommand(self.api, self.db)
            else:
                incomplete_leagues = LeagueSyncCommand(self.api, self.db).get_incomplete_leagues()
                if incomplete_leagues:
                    logging.info("Auto mode: Running league sync (low activity hours)")
                    command = LeagueSyncCommand(self.api, self.db)
                else:
                    logging.info("Auto mode: No action needed (low activity hours)")
                    return True

            return command.execute()

        except Exception as e:
            logging.error(f"Auto command failed: {e}")
            return False

    def _has_live_games(self) -> bool:
        live_fixtures = self.db.get_live_fixtures()
        return len(live_fixtures) > 0


# ---------------------------------------------------------------------------
# Command factory
# ---------------------------------------------------------------------------

class CommandFactory:
    @staticmethod
    def create_command(command_type: CommandType, api_client: APIClient, db_service: DatabaseService) -> Command:
        commands = {
            CommandType.FULL_LOAD: FullLoadCommand,
            CommandType.INCREMENTAL: IncrementalUpdateCommand,
            CommandType.LEAGUE_SYNC: LeagueSyncCommand,
            CommandType.ODDS_SYNC: OddsSyncCommand,
            CommandType.AUTO: AutoCommand
        }
        command_class = commands.get(command_type)
        if not command_class:
            raise ValueError(f"Unknown command type: {command_type}")
        return command_class(api_client, db_service)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class FootballProcessor:
    def __init__(self):
        self.api_client = APIClient(AF_API_KEY, AF_BASE_URL)
        self.db_service = DatabaseService(create_client(SUPABASE_URL, SUPABASE_ANON_KEY))

    def run(self, command_type: CommandType = CommandType.AUTO) -> bool:
        start_time = datetime.now()
        try:
            if not self._should_execute(command_type):
                logging.info(f"Skipping execution - recent run detected for {command_type.value}")
                return True

            command = CommandFactory.create_command(command_type, self.api_client, self.db_service)
            logging.info(f"Executing: {command.get_description()}")

            success = command.execute()

            duration = datetime.now() - start_time
            requests_used = REQUEST_TRACKER.request_count

            if success:
                logging.info(f"Command completed successfully in {duration}")
                logging.info(
                    f"API requests used: {requests_used}/{REQUEST_LIMIT_DAILY} ({REQUEST_TRACKER.get_usage_percentage():.1f}%)")
                self._log_execution_success(command_type, duration, requests_used)
            else:
                logging.error(f"Command failed after {duration}")
                self._log_execution_failure(command_type, duration, requests_used)

            return success

        except Exception as e:
            duration = datetime.now() - start_time
            logging.error(f"Command execution failed after {duration}: {e}")
            self._log_execution_failure(command_type, duration, REQUEST_TRACKER.request_count, str(e))
            return False

    def _should_execute(self, command_type: CommandType) -> bool:
        try:
            cooldown_map = {
                CommandType.FULL_LOAD: 6 * 60,
                CommandType.INCREMENTAL: 5,
                CommandType.LEAGUE_SYNC: 60,
                CommandType.ODDS_SYNC: 10,
                CommandType.AUTO: 10
            }
            cooldown_minutes = cooldown_map.get(command_type, 10)
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)).isoformat()
            try:
                result = self.db_service.client.table("execution_log").select("id").eq(
                    "command_type", command_type.value
                ).eq("status", "success").gte("created_at", cutoff).limit(1).execute()
                if result.data:
                    logging.info(f"Recent successful execution found for {command_type.value}")
                    return False
            except Exception:
                pass
            return True
        except Exception as e:
            logging.warning(f"Failed to check execution history: {e}")
            return True

    def _log_execution_start(self, command_type: CommandType):
        logging.debug(f"Skipping execution_log start insert for {command_type.value}")

    def _log_execution_success(self, command_type: CommandType, duration: timedelta, requests_used: int):
        logging.debug(f"Skipping execution_log success insert for {command_type.value}")

    def _log_execution_failure(self, command_type: CommandType, duration: timedelta, requests_used: int,
                               error: str = None):
        logging.debug(f"Skipping execution_log failure insert for {command_type.value}: {error}")

    def get_system_status(self) -> Dict[str, Any]:
        try:
            status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_usage": {
                    "requests_used": REQUEST_TRACKER.request_count,
                    "daily_limit": REQUEST_LIMIT_DAILY,
                    "usage_percentage": REQUEST_TRACKER.get_usage_percentage()
                }
            }
            try:
                fixtures_result = self.db_service.client.table("fixtures").select("id", count="exact").execute()
                status["data_counts"] = {"fixtures": fixtures_result.count}
            except:
                pass
            try:
                live_fixtures = self.db_service.get_live_fixtures()
                upcoming_fixtures = self.db_service.get_upcoming_fixtures(24)
                status["current_activity"] = {
                    "live_games": len(live_fixtures),
                    "upcoming_24h": len(upcoming_fixtures)
                }
            except:
                pass
            try:
                result = self.db_service.client.table("execution_log").select(
                    "command_type,status,completed_at,duration_seconds,api_requests_used"
                ).order("completed_at", desc=True).limit(5).execute()
                status["recent_executions"] = result.data
            except:
                pass
            return status
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Football Data Processor - Intelligent Data Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python football_processor.py                    # Auto mode (intelligent decision)
  python football_processor.py auto               # Same as above
  python football_processor.py full               # Full data load
  python football_processor.py incremental        # Update active games + smart odds + predictions (missing only)
  python football_processor.py leagues            # Sync league data
  python football_processor.py odds               # Smart odds sync (missing/stale only)
  python football_processor.py --status           # Show system status
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="auto",
        choices=["full", "incremental", "leagues", "odds", "auto"],
        help="Command to execute (default: auto)"
    )

    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--force", action="store_true", help="Force execution ignoring recent runs")

    args = parser.parse_args()
    processor = FootballProcessor()

    if args.status:
        status = processor.get_system_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if args.force:
        processor._should_execute = lambda x: True

    command_map = {
        "full": CommandType.FULL_LOAD,
        "incremental": CommandType.INCREMENTAL,
        "leagues": CommandType.LEAGUE_SYNC,
        "odds": CommandType.ODDS_SYNC,
        "auto": CommandType.AUTO
    }

    command_type = command_map[args.command]
    logging.info(f"Starting Football Data Processor - Command: {command_type.value}")
    success = processor.run(command_type)
    if success:
        logging.info("Execution completed successfully")
    else:
        logging.error("Execution failed")
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
