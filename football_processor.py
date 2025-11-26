import argparse
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import requests
from decouple import config
from pydantic import BaseModel, Field
from slugify import slugify
from supabase import create_client, Client

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
LOG_LEVEL = config("LOG_LEVEL", default="INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(levelname)s - %(message)s")

# Tenta Service Key (Escrita), senão Anon (Leitura)
SUPABASE_URL = config("SUPABASE_URL")
SUPABASE_KEY = config("SUPABASE_SERVICE_KEY", default=config("SUPABASE_ANON_KEY", default=None))

if not SUPABASE_KEY:
    raise ValueError("Chave do Supabase não encontrada. Configure SUPABASE_SERVICE_KEY no .env")

AF_API_KEY = config("AF_API_KEY")
AF_BASE_URL = config("AF_BASE_URL", default="https://v3.football.api-sports.io")

DEFAULT_TZ = config("DEFAULT_TIMEZONE", default="America/Sao_Paulo")
SEASON = int(config("AF_SEASON", default="2025"))
MAX_IDS_PER_BATCH = int(config("MAX_IDS_PER_BATCH", default=20))


# ==============================================================================
# MODELOS (PYDANTIC)
# ==============================================================================

class FixtureStatus(str, Enum):
    NS = "NS"
    FT = "FT"
    LIVE = "LIVE"
    HT = "HT"

    @classmethod
    def live(cls) -> List[str]:
        return ["1H", "HT", "2H", "ET", "BT", "LIVE", "PEN", "SUSP", "INT"]

    @classmethod
    def finished(cls) -> List[str]:
        return ["FT", "AET", "PEN", "ABD", "CANC", "AWD", "WO"]


class League(BaseModel):
    id: int
    name: str
    type: Optional[str] = None
    country: Optional[str] = None
    logo: Optional[str] = None
    season: int
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Team(BaseModel):
    id: int
    name: str
    code: Optional[str] = None
    country: Optional[str] = None
    founded: Optional[int] = None
    logo: Optional[str] = None
    venue_id: Optional[int] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_capacity: Optional[int] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Player(BaseModel):
    id: int
    name: str
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    age: Optional[int] = None
    nationality: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    photo: Optional[str] = None
    injured: Optional[bool] = False
    current_team_id: Optional[int] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Fixture(BaseModel):
    id: int
    referee: Optional[str] = None
    timezone: Optional[str] = None
    date: Optional[datetime] = None
    timestamp: Optional[int] = None
    status_long: Optional[str] = None
    status_short: Optional[str] = None
    elapsed: Optional[int] = None
    league_id: int
    season: int
    round: Optional[str] = None
    home_team_id: int
    away_team_id: int
    goals_home: Optional[int] = None
    goals_away: Optional[int] = None
    score_detailed: Optional[Dict[str, Any]] = None
    slug: str
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FixtureEvent(BaseModel):
    fixture_id: int
    time_elapsed: Optional[int] = None
    time_extra: Optional[int] = None
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    assist_id: Optional[int] = None
    type: Optional[str] = None
    detail: Optional[str] = None
    comments: Optional[str] = None


class FixtureLineup(BaseModel):
    fixture_id: int
    team_id: int
    coach_id: Optional[int] = None
    coach_name: Optional[str] = None
    formation: Optional[str] = None


class FixtureLineupPlayer(BaseModel):
    fixture_id: int
    team_id: int
    player_id: int
    number: Optional[int] = None
    pos: Optional[str] = None
    grid: Optional[str] = None
    type: str


class FixturePlayerStats(BaseModel):
    fixture_id: int
    team_id: int
    player_id: int
    minutes: Optional[int] = None
    rating: Optional[str] = None
    goals_total: Optional[int] = None
    assists: Optional[int] = None
    shots_total: Optional[int] = None
    shots_on: Optional[int] = None
    passes_total: Optional[int] = None
    passes_key: Optional[int] = None
    passes_accuracy: Optional[str] = None
    tackles_total: Optional[int] = None
    cards_yellow: Optional[int] = None
    cards_red: Optional[int] = None


class FixtureInjury(BaseModel):
    fixture_id: int
    team_id: int
    player_id: int
    reason: Optional[str] = None
    type: Optional[str] = None


class FixtureStatistic(BaseModel):
    fixture_id: int
    team_id: int
    type: str
    value: Optional[str] = None


class Prediction(BaseModel):
    fixture_id: int
    prob_home: Optional[float] = None
    prob_draw: Optional[float] = None
    prob_away: Optional[float] = None
    advice: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Odds(BaseModel):
    fixture_id: int
    bookmaker_name: Optional[str] = None
    market_name: Optional[str] = None
    value: Optional[str] = None
    odd: Optional[float] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Standing(BaseModel):
    league_id: int
    season: int
    team_id: int
    rank: int
    points: Optional[int] = None
    goals_diff: Optional[int] = None
    group_name: Optional[str] = None
    form: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    all_played: Optional[int] = None
    all_win: Optional[int] = None
    all_draw: Optional[int] = None
    all_lose: Optional[int] = None
    all_goals_for: Optional[int] = None
    all_goals_against: Optional[int] = None
    home_played: Optional[int] = None
    home_win: Optional[int] = None
    home_draw: Optional[int] = None
    home_lose: Optional[int] = None
    home_goals_for: Optional[int] = None
    home_goals_against: Optional[int] = None
    away_played: Optional[int] = None
    away_win: Optional[int] = None
    away_draw: Optional[int] = None
    away_lose: Optional[int] = None
    away_goals_for: Optional[int] = None
    away_goals_against: Optional[int] = None
    update_time: datetime


# ==============================================================================
# SERVIÇOS
# ==============================================================================

class APIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": api_key, "Accept": "application/json"})

    def get(self, path: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            time.sleep(0.15)
            return response.json()
        except requests.RequestException as e:
            logging.error(f"API Error {url}: {e}")
            return {}


class DatabaseService:
    def __init__(self, client: Client):
        self.client = client

    def get_active_leagues(self) -> List[int]:
        res = self.client.table("leagues").select("id").execute()
        return [r["id"] for r in res.data]

    def get_active_teams(self) -> List[int]:
        res = self.client.table("teams").select("id").execute()
        return [r["id"] for r in res.data]

    def get_fixture_ids(self, status_in: List[str] = None, hours_lookback: int = None, hours_lookahead: int = None) -> \
    List[int]:
        query = self.client.table("fixtures").select("id")
        now = datetime.now(timezone.utc)
        if status_in:
            query = query.in_("status_short", status_in)
        if hours_lookback is not None:
            start = (now - timedelta(hours=hours_lookback)).isoformat()
            query = query.gte("date", start)
        if hours_lookahead is not None:
            end = (now + timedelta(hours=hours_lookahead)).isoformat()
            query = query.lte("date", end)
        res = query.execute()
        return [r["id"] for r in res.data]

    def get_ids_missing_predictions(self, fixture_ids: List[int]) -> List[int]:
        if not fixture_ids: return []
        res = self.client.table("predictions").select("fixture_id").in_("fixture_id", fixture_ids).execute()
        existing = {r["fixture_id"] for r in res.data}
        return [fid for fid in fixture_ids if fid not in existing]

    def upsert(self, table: str, models: List[BaseModel], on_conflict: str = "id"):
        if not models: return
        data = [m.model_dump(mode='json', exclude_none=False) for m in models]
        for i in range(0, len(data), 1000):
            batch = data[i:i + 1000]
            try:
                self.client.table(table).upsert(batch, on_conflict=on_conflict).execute()
            except Exception as e:
                logging.error(f"Error upserting to {table}: {e}")


# ==============================================================================
# COMANDOS
# ==============================================================================

class Command(ABC):
    def __init__(self, api: APIClient, db: DatabaseService):
        self.api = api
        self.db = db

    @abstractmethod
    def execute(self) -> bool:
        pass


class LeagueSyncCommand(Command):
    def execute(self) -> bool:
        logging.info("Syncing Major Leagues...")
        targets = [
            "Brazil", "England", "Spain", "France", "Italy", "Germany",
            "Portugal", "Netherlands", "Argentina", "Europe", "South America", "World"
        ]
        leagues_map = {}

        vip_leagues_ids = [
            39, 40, 45, 48, 140, 141, 143, 135, 136, 137, 78, 79, 81, 61, 62, 66,
            94, 88, 128, 144, 2, 3, 848, 13, 11, 15, 9, 1, 4
        ]

        for lid in vip_leagues_ids:
            try:
                data = self.api.get("leagues", {"id": lid, "current": "true"})
                for item in data.get("response", []):
                    l_data, c_data = item["league"], item["country"]
                    leagues_map[l_data["id"]] = League(
                        id=l_data["id"], name=l_data["name"], type=l_data["type"],
                        country=c_data["name"], logo=l_data["logo"], season=l_data.get("season") or SEASON
                    )
            except Exception:
                continue

        try:
            data = self.api.get("leagues", {"country": "Brazil", "current": "true"})
            for item in data.get("response", []):
                l_data, c_data = item["league"], item["country"]
                name = l_data["name"]
                ignore_terms = ["U20", "U19", "U17", "Women", "Feminino", "A2", "A3", "A4", "2", "3", "Amapaense",
                                "Rondoniense", "Tocantinense", "Acreano"]
                must_have = ["Serie A", "Serie B", "Serie C", "Serie D", "Copa do Brasil", "Copa do Nordeste",
                             "Supercopa"]

                is_vip = any(k in name for k in must_have)
                is_bad = any(k in name for k in ignore_terms)

                if is_vip or (not is_bad):
                    leagues_map[l_data["id"]] = League(
                        id=l_data["id"], name=l_data["name"], type=l_data["type"],
                        country=c_data["name"], logo=l_data["logo"], season=l_data.get("season") or SEASON
                    )
        except Exception as e:
            logging.error(f"Error syncing Brazil: {e}")

        if leagues_map:
            self.db.upsert("leagues", list(leagues_map.values()))
            logging.info(f"Synced {len(leagues_map)} relevant leagues.")
        return True


class SquadSyncCommand(Command):
    def execute(self) -> bool:
        logging.info("Syncing Squads (Players)...")
        team_ids = self.db.get_active_teams()
        if not team_ids:
            logging.warning("No teams found in DB. Run Full Load first.")
            return False

        for tid in team_ids:
            try:
                data = self.api.get("players/squads", {"team": tid})
                players_map = {}
                for item in data.get("response", []):
                    for p in item.get("players", []):
                        players_map[p["id"]] = Player(
                            id=p["id"], name=p["name"], age=p.get("age"),
                            number=p.get("number"), pos=p.get("position"),
                            photo=p.get("photo"), current_team_id=tid
                        )
                if players_map:
                    self.db.upsert("players", list(players_map.values()))
                    logging.info(f"Synced {len(players_map)} players for team {tid}")
            except Exception as e:
                logging.error(f"Error syncing squad for team {tid}: {e}")
                continue
        return True


class StandingsSyncCommand(Command):
    def execute(self) -> bool:
        logging.info("Syncing Standings...")
        leagues = self.db.get_active_leagues()
        standings_map = {}
        teams_map = {}

        for lid in leagues:
            try:
                data = self.api.get("standings", {"league": lid, "season": SEASON})
                for item in data.get("response", []):
                    l_id, season = item["league"]["id"], item["league"]["season"]
                    for group in item["league"]["standings"]:
                        for row in group:
                            tid = row["team"].get("id")
                            if not tid: continue
                            teams_map[tid] = Team(id=tid, name=row["team"]["name"], logo=row["team"]["logo"])

                            key = (l_id, season, tid)
                            standings_map[key] = Standing(
                                league_id=l_id, season=season, team_id=tid,
                                rank=row["rank"], points=row["points"], goals_diff=row["goalsDiff"],
                                group_name=row["group"], form=row["form"], status=row["status"],
                                description=row["description"], all_played=row["all"]["played"],
                                all_win=row["all"]["win"], all_draw=row["all"]["draw"], all_lose=row["all"]["lose"],
                                all_goals_for=row["all"]["goals"]["for"],
                                all_goals_against=row["all"]["goals"]["against"],
                                home_played=row["home"]["played"], home_win=row["home"]["win"],
                                home_draw=row["home"]["draw"], home_lose=row["home"]["lose"],
                                home_goals_for=row["home"]["goals"]["for"],
                                home_goals_against=row["home"]["goals"]["against"],
                                away_played=row["away"]["played"], away_win=row["away"]["win"],
                                away_draw=row["away"]["draw"], away_lose=row["away"]["lose"],
                                away_goals_for=row["away"]["goals"]["for"],
                                away_goals_against=row["away"]["goals"]["against"],
                                update_time=datetime.fromisoformat(row["update"].replace("Z", "+00:00"))
                            )
            except Exception:
                continue

        if teams_map: self.db.upsert("teams", list(teams_map.values()))
        if standings_map: self.db.upsert("standings", list(standings_map.values()),
                                         on_conflict="league_id,season,team_id")
        return True


class IncrementalUpdateCommand(Command):
    def execute(self) -> bool:
        logging.info("Running Incremental Update...")
        live = self.db.get_fixture_ids(status_in=FixtureStatus.live())
        recent = self.db.get_fixture_ids(status_in=FixtureStatus.finished(), hours_lookback=6)
        upcoming = self.db.get_fixture_ids(hours_lookahead=24)

        priority = list(set(live + recent + upcoming))
        if priority: self._process_fixtures(priority)

        missing_pred = self.db.get_ids_missing_predictions(upcoming)
        if missing_pred:
            self._fetch_predictions(missing_pred)
            self._fetch_injuries(missing_pred)
        return True

    def _process_fixtures(self, fixture_ids: List[int]):
        for i in range(0, len(fixture_ids), MAX_IDS_PER_BATCH):
            batch = fixture_ids[i:i + MAX_IDS_PER_BATCH]
            data = self.api.get("fixtures", {"ids": "-".join(map(str, batch)), "timezone": DEFAULT_TZ})
            self._parse_and_save(data.get("response", []))

    def _parse_and_save(self, response: List[Dict[str, Any]]):
        fixtures, teams, players = [], {}, {}
        lineups_map, lineup_players_map = {}, {}
        events, stats_map, player_stats_map = [], {}, {}

        for item in response:
            f, l, t, g, s = item["fixture"], item["league"], item["teams"], item["goals"], item["score"]

            for side in ["home", "away"]:
                tid = t[side]["id"]
                teams[tid] = Team(id=tid, name=t[side]["name"], logo=t[side]["logo"])

            dt_utc = datetime.fromisoformat(f["date"].replace("Z", "+00:00")) if f["date"] else None
            slug = slugify(f"{t['home']['name']} x {t['away']['name']} {f['id']}")

            fixtures.append(Fixture(
                id=f["id"], referee=f.get("referee"), timezone=f.get("timezone"),
                date=dt_utc, timestamp=f.get("timestamp"), status_long=f["status"].get("long"),
                status_short=f["status"].get("short"), elapsed=f["status"].get("elapsed"),
                league_id=l["id"], season=l["season"], round=l["round"],
                home_team_id=t["home"]["id"], away_team_id=t["away"]["id"],
                goals_home=g.get("home"), goals_away=g.get("away"), score_detailed=s, slug=slug
            ))

            for lineup in item.get("lineups", []):
                tid = lineup["team"]["id"]
                lineups_map[(f["id"], tid)] = FixtureLineup(
                    fixture_id=f["id"], team_id=tid, coach_id=lineup.get("coach", {}).get("id"),
                    coach_name=lineup.get("coach", {}).get("name"), formation=lineup.get("formation")
                )
                for section, p_type in [("startXI", "XI"), ("substitutes", "SUB")]:
                    for wrapper in lineup.get(section, []):
                        p, pid = wrapper["player"], wrapper["player"]["id"]
                        if not pid: continue
                        players[pid] = Player(id=pid, name=p["name"],
                                              photo=f"https://media.api-sports.io/football/players/{pid}.png")
                        lineup_players_map[(f["id"], tid, pid)] = FixtureLineupPlayer(
                            fixture_id=f["id"], team_id=tid, player_id=pid,
                            number=p.get("number"), pos=p.get("pos"), grid=p.get("grid"), type=p_type
                        )

            for ev in item.get("events", []):
                pid, aid = ev["player"].get("id"), ev["assist"].get("id")
                if pid: players[pid] = Player(id=pid, name=ev["player"]["name"],
                                              photo=f"https://media.api-sports.io/football/players/{pid}.png")
                if aid: players[aid] = Player(id=aid, name=ev["assist"]["name"],
                                              photo=f"https://media.api-sports.io/football/players/{aid}.png")
                events.append(FixtureEvent(
                    fixture_id=f["id"], time_elapsed=ev["time"].get("elapsed"), time_extra=ev["time"].get("extra"),
                    team_id=ev["team"].get("id"), player_id=pid, assist_id=aid, type=ev.get("type"),
                    detail=ev.get("detail"), comments=ev.get("comments")
                ))

            for p_team in item.get("players", []):
                tid = p_team["team"]["id"]
                for p_data in p_team["players"]:
                    p, pid, st = p_data["player"], p_data["player"]["id"], p_data["statistics"][0]
                    if not pid: continue
                    players[pid] = Player(id=pid, name=p["name"], firstname=p.get("firstname"),
                                          lastname=p.get("lastname"), photo=p.get("photo"))
                    player_stats_map[(f["id"], tid, pid)] = FixturePlayerStats(
                        fixture_id=f["id"], team_id=tid, player_id=pid,
                        minutes=st["games"]["minutes"], rating=st["games"]["rating"],
                        goals_total=st["goals"]["total"], assists=st["goals"]["assists"],
                        shots_total=st["shots"]["total"], shots_on=st["shots"]["on"],
                        passes_total=st["passes"]["total"], passes_key=st["passes"]["key"],
                        passes_accuracy=st["passes"]["accuracy"],
                        tackles_total=st["tackles"]["total"], cards_yellow=st["cards"]["yellow"],
                        cards_red=st["cards"]["red"]
                    )

            for st_team in item.get("statistics", []):
                tid = st_team["team"]["id"]
                for st in st_team.get("statistics", []):
                    if st["value"] is not None:
                        stats_map[(f["id"], tid, st["type"])] = FixtureStatistic(fixture_id=f["id"], team_id=tid,
                                                                                 type=st["type"],
                                                                                 value=str(st["value"]))

        if teams: self.db.upsert("teams", list(teams.values()))
        if players: self.db.upsert("players", list(players.values()))
        if fixtures: self.db.upsert("fixtures", fixtures)

        if lineups_map: self.db.upsert("fixture_lineups", list(lineups_map.values()), "fixture_id,team_id")
        if lineup_players_map: self.db.upsert("fixture_lineup_players", list(lineup_players_map.values()),
                                              "fixture_id,team_id,player_id")
        if events: self.db.upsert("fixture_events", events, "id")
        if stats_map: self.db.upsert("fixture_statistics", list(stats_map.values()), "fixture_id,team_id,type")
        if player_stats_map: self.db.upsert("fixture_player_stats", list(player_stats_map.values()),
                                            "fixture_id,team_id,player_id")

    def _fetch_predictions(self, fixture_ids: List[int]):
        preds = []
        for fid in fixture_ids:
            try:
                data = self.api.get("predictions", {"fixture": fid})
                for item in data.get("response", []):
                    p = item["predictions"]
                    comp = item.get("comparison", {}).get("total", {})

                    # LÓGICA HÍBRIDA: EMPATE (API) + FORÇA RELATIVA (COMPARISON)
                    raw_draw = p.get("percent", {}).get("draw")
                    prob_draw = float(raw_draw.strip("%")) / 100 if raw_draw else 0.25

                    if comp.get("home") and comp.get("away"):
                        comp_home = float(comp["home"].strip("%")) / 100
                        comp_away = float(comp["away"].strip("%")) / 100
                        total_comp = comp_home + comp_away
                        if total_comp == 0: total_comp = 1

                        remaining_prob = 1.0 - prob_draw
                        prob_home = (comp_home / total_comp) * remaining_prob
                        prob_away = (comp_away / total_comp) * remaining_prob
                    else:
                        raw_home = p.get("percent", {}).get("home")
                        raw_away = p.get("percent", {}).get("away")
                        prob_home = float(raw_home.strip("%")) / 100 if raw_home else 0.375
                        prob_away = float(raw_away.strip("%")) / 100 if raw_away else 0.375

                    preds.append(Prediction(
                        fixture_id=fid,
                        prob_home=round(prob_home, 4),
                        prob_draw=round(prob_draw, 4),
                        prob_away=round(prob_away, 4),
                        advice=p.get("advice"), raw_json=item
                    ))
            except Exception as e:
                logging.error(f"Error prediction {fid}: {e}")
                continue

        if preds:
            self.db.upsert("predictions", preds, "fixture_id")

    def _fetch_injuries(self, fixture_ids: List[int]):
        injuries_map = {}
        for fid in fixture_ids:
            try:
                data = self.api.get("injuries", {"fixture": fid})
                for item in data.get("response", []):
                    p = item["player"]
                    if not p.get("id"): continue
                    key = (fid, p["id"])
                    injuries_map[key] = FixtureInjury(
                        fixture_id=fid, team_id=item["team"]["id"], player_id=p["id"],
                        reason=p["reason"], type=p["type"]
                    )
            except Exception:
                continue

        if injuries_map:
            self.db.upsert("fixture_injuries", list(injuries_map.values()), "fixture_id,player_id")


class OddsSyncCommand(Command):
    def execute(self) -> bool:
        logging.info("Syncing Odds (All Bookmakers)...")
        upcoming = self.db.get_fixture_ids(status_in=["NS"], hours_lookahead=48)
        if not upcoming:
            logging.info("No upcoming fixtures found.")
            return True

        logging.info(f"Scanning odds for {len(upcoming)} fixtures...")

        all_odds = []
        successful_fids = set()

        for fid in upcoming:
            try:
                data = self.api.get("odds", {"fixture": fid})
                response = data.get("response", [])
                if not response: continue

                item = response[0]
                successful_fids.add(fid)

                # AGORA PEGAMOS TODAS AS ODDS DISPONÍVEIS, NÃO APENAS UMA
                bookmakers = item.get("bookmakers", [])
                for bm in bookmakers:
                    for bet in bm.get("bets", []):
                        if bet["name"] in ["Match Winner", "Both Teams To Score", "Goals Over/Under"]:
                            for val in bet.get("values", []):
                                all_odds.append(Odds(
                                    fixture_id=fid,
                                    bookmaker_name=bm["name"],
                                    market_name=bet["name"],
                                    value=str(val["value"]),
                                    odd=float(val["odd"])
                                ))
            except Exception as e:
                logging.error(f"Error odds {fid}: {e}")
                continue

        if all_odds:
            # Limpa e insere (refresh completo para evitar duplicatas e odds velhas)
            if successful_fids:
                self.db.client.table("odds").delete().in_("fixture_id", list(successful_fids)).execute()
            self.db.upsert("odds", all_odds, "id")
            logging.info(f"Synced {len(all_odds)} odds.")

        return True


class FullLoadCommand(Command):
    def execute(self) -> bool:
        logging.info("Starting Full Load (Economic Mode)...")
        LeagueSyncCommand(self.api, self.db).execute()
        StandingsSyncCommand(self.api, self.db).execute()
        leagues = self.db.get_active_leagues()
        now = datetime.now()
        start_day, end_day = 0, 2

        for league_id in leagues:
            for day_offset in range(start_day, end_day):
                date_str = (now + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                try:
                    data = self.api.get("fixtures", {"date": date_str, "league": league_id, "season": SEASON})
                    response = data.get("response", [])
                    if response:
                        IncrementalUpdateCommand(self.api, self.db)._parse_and_save(response)
                        logging.info(f"League {league_id} on {date_str}: Found {len(response)} fixtures.")
                except Exception:
                    continue
        return True


class FootballProcessor:
    def __init__(self):
        self.api = APIClient(AF_API_KEY, AF_BASE_URL)
        self.db = DatabaseService(create_client(SUPABASE_URL, SUPABASE_KEY))

    def run(self, command_name: str):
        commands = {
            "full": FullLoadCommand,
            "incremental": IncrementalUpdateCommand,
            "leagues": LeagueSyncCommand,
            "standings": StandingsSyncCommand,
            "odds": OddsSyncCommand,
            "squads": SquadSyncCommand
        }
        if command_name == "auto":
            hour = datetime.now().hour
            command_name = "full" if hour == 3 else "incremental"
        cmd = commands.get(command_name)
        if cmd: cmd(self.api, self.db).execute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="auto")
    args = parser.parse_args()
    FootballProcessor().run(args.command)


if __name__ == "__main__":
    main()