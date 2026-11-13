from typing import List, Optional, Any, Dict
from espn_api.basketball import League
from datetime import date
import os
from dotenv import load_dotenv
import pandas as pd
import requests
from pydantic import BaseModel, Field

load_dotenv()

SWID = os.getenv("SWID", "")
ESPN_S2 = os.getenv("ESPN_S2", "")

class StatBlock(BaseModel):
    gp: int = 0
    min: float = 0.0
    pts: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    fg3m: float = 0.0
    to: float = 0.0
    fgm: float = 0.0
    fga: float = 0.0
    ftm: float = 0.0
    fta: float = 0.0
    total_fpts: float = 0.0
    fppg: float = 0.0

    @property
    def fg_pct(self) -> float:
        return round(self.fgm / self.fga, 3) if self.fga > 0 else 0.0

    @property
    def ft_pct(self) -> float:
        return round(self.ftm / self.fta, 3) if self.fta > 0 else 0.0

class Player(BaseModel):
    id: int
    name: str
    nba_team: str
    positions: str
    headshot_url: Optional[str] = None
    team_logo_url: Optional[str] = None
    injury_status: Optional[str] = "HEALTHY"
    is_bookmarked: bool = False 
    fantasy_team_id: Optional[int] = None
    fantasy_team_name: Optional[str] = None
    eligible_slots: List[str] = Field(default_factory=list)
    expected_return_date: Optional[date] = None
    injured: bool = False
    schedule: Dict[str, Any] = Field(default_factory=dict)
    
    gp: int = 0
    min: float = 0.0
    pts: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    fg3m: float = 0.0
    to: float = 0.0
    fgm: float = 0.0
    fga: float = 0.0
    ftm: float = 0.0
    fta: float = 0.0
    total_fpts: float = 0.0
    fppg: float = 0.0

    timeframe_stats: Dict[str, StatBlock] = Field(default_factory=dict)

class Team(BaseModel):
    id: int
    name: str
    abbrev: str
    owner: str
    roster_count: int

def fetch_espn_teams(league_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> List[Team]:
    try:
        season_year = 2027
        
        if is_public:
            nba_league = League(league_id=league_id, year=season_year)
        else:
            nba_league = League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)

        teams = []
        for t in nba_league.teams:
            teams.append(
                Team(
                    id=t.team_id,
                    name=t.team_name,
                    abbrev=getattr(t, 'team_abbrev', ''),
                    owner=getattr(t, 'owner', 'Unknown'),
                    roster_count=len(t.roster) if hasattr(t, 'roster') else 0
                )
            )
        return teams

    except Exception as e:
        print(e)
        return []

def fetch_nba_news(limit: int = 20) -> dict:
    response = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news",
        timeout=10,
    )
    response.raise_for_status()
    news = response.json()
    if not isinstance(news, dict) or not isinstance(news.get("articles"), list):
        raise ValueError("ESPN returned an unexpected NBA news response")
    news["articles"] = news["articles"][:limit]
    return news

def extract_stat_block(timeframe_dict: dict) -> StatBlock:
    if not isinstance(timeframe_dict, dict) or not timeframe_dict:
        return StatBlock()

    avg_stats = timeframe_dict.get('avg')
    has_averages = isinstance(avg_stats, dict) and bool(avg_stats)
    if not has_averages:
        total_stats = timeframe_dict.get('total', {})
        avg_stats = total_stats if isinstance(total_stats, dict) else {}
    games_played = int(avg_stats.get('GP', 0))
    divisor = 1 if has_averages or games_played == 0 else games_played
    fppg = float(timeframe_dict.get('applied_avg', 0.0))
    total_fpts = float(timeframe_dict.get('applied_total', 0.0))

    def average_stat(key: str, alias: str = "", default: float = 0.0) -> float:
        value = avg_stats.get(key, avg_stats.get(alias, default))
        return round(float(value) / divisor, 2)

    return StatBlock(
        gp=games_played,
        min=average_stat('MIN', 'MPG'),
        pts=average_stat('PTS', 'PPG'),
        reb=average_stat('REB', 'RPG'),
        ast=average_stat('AST', 'APG'),
        stl=average_stat('STL', 'SPG'),
        blk=average_stat('BLK', 'BPG'),
        fg3m=average_stat('3PM', '3PG'),
        to=average_stat('TO', 'TOPG'),
        fgm=average_stat('FGM'),
        fga=average_stat('FGA'),
        ftm=average_stat('FTM'),
        fta=average_stat('FTA'),
        total_fpts=round(total_fpts, 2),
        fppg=round(fppg if fppg > 0 else average_stat('PTS', 'PPG'), 2)
    )

def parse_player_data(p, current_year: int = 2027) -> Player:
    def value(name: str, default=None):
        if isinstance(p, dict):
            return p.get(name, default)
        return getattr(p, name, default)

    stats_data = value('stats', {})
    if not isinstance(stats_data, dict):
        stats_data = {}
    
    block_7d = stats_data.get(f'{current_year}_last_7', {})
    block_30d = stats_data.get(f'{current_year}_last_30', {})
    block_total = stats_data.get(f'{current_year}_total', {})
    block_proj = stats_data.get(f'{current_year}_projected', {})

    timeframes = {
        "7d": extract_stat_block(block_7d),
        "30d": extract_stat_block(block_30d),
        f"{current_year}": extract_stat_block(block_total),
        "proj": extract_stat_block(block_proj)
    }

    base = timeframes[f"{current_year}"]
    player_id = int(value('playerId', 0))
    nba_team = value('proTeam', 'N/A')
    injury_status = value('injuryStatus', "HEALTHY")
    if not isinstance(injury_status, str):
        injury_status = None
    team_abbrev = nba_team.upper() if isinstance(nba_team, str) else ''

    return Player(
        id=player_id,
        name=value('name', 'Unknown Player'),
        nba_team=nba_team,
        headshot_url=f"https://a.espncdn.com/i/headshots/nba/players/full/{player_id}.png",
        team_logo_url=(
            f"https://a.espncdn.com/i/teamlogos/nba/500/{team_abbrev}.png"
            if team_abbrev and team_abbrev != "N/A"
            else None
        ),
        positions=value('position', value('positions', "")),
        injury_status=injury_status,
        eligible_slots=value('eligibleSlots', []),
        expected_return_date=value('expected_return_date'),
        injured=bool(value('injured', False)),
        schedule=value('schedule', {}),
        gp=base.gp,
        min=base.min,
        pts=base.pts,
        reb=base.reb,
        ast=base.ast,
        stl=base.stl,
        blk=base.blk,
        fg3m=base.fg3m,
        to=base.to,
        fgm=base.fgm,
        fga=base.fga,
        ftm=base.ftm,
        fta=base.fta,
        total_fpts=float(value('total_points', base.total_fpts)),
        fppg=float(value('avg_points', base.fppg)),
        timeframe_stats=timeframes
    )

def fetch_espn_players(league_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> List[Player]:
    try:
        season_year = 2027
        
        if is_public:
            nba_league = League(league_id=league_id, year=season_year)
        else:
            nba_league = League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)

        player_pool = nba_league.free_agents(size=25)
        
        return [parse_player_data(p, current_year=season_year) for p in player_pool]

    except Exception as e:
        print(e)
        return []

def fetch_team_roster(league_id: int, team_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> List[Player]:
    try:
        season_year = 2027
        
        if is_public:
            nba_league = League(league_id=league_id, year=season_year)
        else:
            nba_league = League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)

        target_team = next((t for t in nba_league.teams if int(t.team_id) == int(team_id)), None)
        
        if not target_team:
            print(f"DEBUG: No team found matching team_id={team_id} among {[t.team_id for t in nba_league.teams]}")
            return []

        return [parse_player_data(p, current_year=season_year) for p in target_team.roster]

    except Exception as e:
        print(e)
        return []

def filter_and_sort_players(
    players: List[Player],
    sort_by: str = "fppg",
    ascending: bool = False,
    only_bookmarked: bool = False,
    timeframe: Optional[str] = None
) -> List[dict]:
    try:
        if not players:
            return []

        data = []
        for p in players:
            p_dict = p.model_dump()
            if timeframe and timeframe in p.timeframe_stats:
                tf_data = p.timeframe_stats[timeframe].model_dump()
                p_dict.update(tf_data)
            data.append(p_dict)

        df = pd.DataFrame(data)

        if only_bookmarked and 'is_bookmarked' in df.columns:
            df = df[df['is_bookmarked'] == True]

        valid_sort_cols = [
            'total_fpts', 'fppg', 'min', 'fgm', 'fga', 'ftm',
            'fta', 'fg3m', 'reb', 'ast', 'stl', 'blk', 'to', 'pts'
        ]

        target_col = sort_by if sort_by in valid_sort_cols else 'fppg'

        if target_col in df.columns:
            df = df.sort_values(by=target_col, ascending=ascending)

        return df.to_dict(orient='records')
    except Exception as e:
        print(e)
        raise

def _team_summary(team, include_roster: bool = True) -> dict:
    summary = {
        "id": team.team_id,
        "name": team.team_name,
        "abbrev": team.team_abbrev,
        "wins": team.wins,
        "losses": team.losses,
        "ties": getattr(team, "ties", 0),
        "points_for": team.points_for,
        "points_against": getattr(team, "points_against", 0.0),
        "standing": getattr(team, "standing", None),
    }
    if include_roster:
        summary["roster"] = [parse_player_data(p).model_dump() for p in team.roster]
    return summary

def _league_settings(league) -> dict:
    settings = league.settings
    return {
        "name": settings.name,
        "scoring_type": settings.scoring_type,
        "scoring_settings": settings._raw_scoring_settings,
        "schedule_settings": settings._raw_schedule_settings,
        "team_count": settings.team_count,
        "trade_deadline": settings.trade_deadline,
        "acquisition_limit": settings.acquisition_limit,
        "matchup_acquisition_limit": settings.matchup_acquisition_limit,
    }

def _serialize_matchup(matchup) -> dict:
    def team_info(team):
        return {
            "id": getattr(team, "team_id", team),
            "name": getattr(team, "team_name", None),
        }

    return {
        "home_team": team_info(matchup.home_team),
        "away_team": team_info(matchup.away_team),
        "winner": matchup.winner,
        "home_score": getattr(matchup, "home_score", None),
        "away_score": getattr(matchup, "away_score", None),
        "home_category_stats": getattr(matchup, "home_stats", None),
        "away_category_stats": getattr(matchup, "away_stats", None),
        "home_category_wins": getattr(matchup, "home_wins", None),
        "away_category_wins": getattr(matchup, "away_wins", None),
    }

def fetch_all_players_search(league_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> List[Player]:
    """Return all rostered players and available free agents."""
    try:
        season_year = 2027
        nba_league = League(league_id=league_id, year=season_year) if is_public else League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)
        
        all_players = {}
        
        for t in nba_league.teams:
            for p in t.roster:
                player_obj = parse_player_data(p, current_year=season_year)
                player_obj.fantasy_team_id = t.team_id
                player_obj.fantasy_team_name = t.team_name
                all_players[player_obj.id] = player_obj

        free_agents = nba_league.free_agents(size=1000)
        for p in free_agents:
            player_obj = parse_player_data(p, current_year=season_year)
            all_players.setdefault(player_obj.id, player_obj)

        return list(all_players.values())
    except Exception as e:
        print(f"Error fetching all search players: {e}")
        return []

def fetch_top_today_performers(league_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> List[dict]:
    """Return the top 25 players by fantasy points scored in today's games."""
    try:
        season_year = 2027
        nba_league = League(league_id=league_id, year=season_year) if is_public else League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)
        performers = {}

        for matchup in nba_league.box_scores(matchup_total=False):
            for team, lineup in (
                (matchup.home_team, matchup.home_lineup),
                (matchup.away_team, matchup.away_lineup),
            ):
                for p in lineup:
                    if not p.points_breakdown and not p.points:
                        continue
                    player = parse_player_data(p, current_year=season_year).model_dump()
                    player.update({
                        "fantasy_team_id": team.team_id,
                        "fantasy_team_name": team.team_name,
                        "today_stats": p.points_breakdown,
                        "today_fantasy_points": p.points,
                        "today_opponent": p.pro_opponent,
                        "lineup_slot": p.slot_position,
                    })
                    performers[p.playerId] = player

        for p in nba_league.free_agents(week=nba_league.current_week, size=1000):
            stat_line = getattr(p, "stats", {}).get(str(nba_league.current_week), {})
            today_stats = stat_line.get("total", {}) or {}
            today_points = float(stat_line.get("applied_total", 0.0))
            if not today_stats and not today_points:
                continue
            player = parse_player_data(p, current_year=season_year).model_dump()
            player.update({
                "today_stats": today_stats,
                "today_fantasy_points": today_points,
            })
            performers.setdefault(p.playerId, player)

        return sorted(
            performers.values(),
            key=lambda player: player["today_fantasy_points"],
            reverse=True,
        )[:25]
    except Exception as e:
        print(f"Error fetching top daily performers: {e}")
        return []

def fetch_trade_evaluator_data(league_id: int, user_team_id: int, target_team_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> dict:
    """Return focused trade context for the two selected teams."""
    try:
        season_year = 2027
        nba_league = League(league_id=league_id, year=season_year) if is_public else League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)
        
        user_team = next((t for t in nba_league.teams if int(t.team_id) == int(user_team_id)), None)
        target_team = next((t for t in nba_league.teams if int(t.team_id) == int(target_team_id)), None)

        if not user_team or not target_team:
            return {"error": "One or both teams not found"}

        return {
            "league_id": league_id,
            "season_year": season_year,
            "current_scoring_period": nba_league.current_week,
            "current_matchup_period": nba_league.currentMatchupPeriod,
            "league_settings": _league_settings(nba_league),
            "user_team": _team_summary(user_team),
            "target_team": _team_summary(target_team),
        }
    except Exception as e:
        print(f"Error evaluating trade rosters: {e}")
        return {}

def fetch_trade_finder_data(league_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> dict:
    try:
        season_year = 2027
        nba_league = League(league_id=league_id, year=season_year) if is_public else League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)
        return {
            "league_id": league_id,
            "season_year": season_year,
            "current_scoring_period": nba_league.current_week,
            "current_matchup_period": nba_league.currentMatchupPeriod,
            "league_settings": _league_settings(nba_league),
            "teams": [_team_summary(team) for team in nba_league.teams],
            "top_free_agents": [
                parse_player_data(player, current_year=season_year).model_dump()
                for player in nba_league.free_agents(size=100)
            ],
        }
    except Exception as e:
        print(f"Error loading trade finder data: {e}")
        return {}

def build_full_ai_context_payload(league_id: int, user_team_id: int, espn_s2: str = "", swid: str = "", is_public: bool = False) -> dict:
    """Endpoint #1 & #5: Assembles full context payload (Matchups, Schedule, Roster, FAs, League Context) for LLM."""
    try:
        season_year = 2027
        nba_league = League(league_id=league_id, year=season_year) if is_public else League(league_id=league_id, year=season_year, espn_s2=espn_s2, swid=swid)

        user_team = next((t for t in nba_league.teams if int(t.team_id) == int(user_team_id)), None)
        if not user_team:
            return {"error": "User team not found"}

        free_agents_raw = nba_league.free_agents(size=100)
        free_agents_parsed = [parse_player_data(p, season_year).model_dump() for p in free_agents_raw]

        current_matchups = [
            _serialize_matchup(matchup)
            for matchup in nba_league.box_scores()
        ]
        user_matchup = next(
            (
                matchup for matchup in current_matchups
                if user_team.team_id in (
                    matchup["home_team"]["id"],
                    matchup["away_team"]["id"],
                )
            ),
            None,
        )

        return {
            "league_id": league_id,
            "season_year": season_year,
            "current_scoring_period": nba_league.current_week,
            "current_matchup_period": nba_league.currentMatchupPeriod,
            "league_settings": _league_settings(nba_league),
            "user_team": _team_summary(user_team),
            "current_matchup": user_matchup,
            "current_matchups": current_matchups,
            "top_free_agents": free_agents_parsed,
            "league_standings": [
                _team_summary(team, include_roster=False)
                for team in nba_league.standings()
            ],
        }
    except Exception as e:
        print(f"Error generating AI payload: {e}")
        return {}