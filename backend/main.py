from fastapi import FastAPI, Query
from fastapi import HTTPException
from typing import Optional
import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from services import (
    fetch_espn_teams,
    fetch_espn_players,
    fetch_team_roster,
    filter_and_sort_players,
    fetch_all_players_search,
    fetch_nba_news,
    fetch_top_today_performers,
    fetch_trade_evaluator_data,
    fetch_trade_finder_data,
    build_full_ai_context_payload,
)
from trade_engine import evaluate_trade, find_trades, find_trades_between_teams

app = FastAPI()

load_dotenv()

SWID = os.getenv("SWID", "")
ESPN_S2 = os.getenv("ESPN_S2", "")
LEAGUE_ID = 1997461255

class TradeEvaluationRequest(BaseModel):
    user_team_id: int
    target_team_id: int
    user_gives: list[int | str]
    target_gives: list[int | str]
    waiver_player: int | str | None = None

@app.get("/api/news")
def get_nba_news(limit: int = Query(default=20, ge=1, le=50)):
    try:
        return fetch_nba_news(limit=limit)
    except (requests.RequestException, ValueError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch ESPN NBA news: {e}",
        ) from e

@app.get("/api/teams")
def get_teams():
    return fetch_espn_teams(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)

@app.get("/api/players/sorted")
def get_sorted_players(
    sort_by: str = "fppg",
    ascending: bool = False,
    only_bookmarked: bool = False,
    timeframe: Optional[str] = "2027"
):
    players = fetch_espn_players(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)
    return filter_and_sort_players(
        players, 
        sort_by=sort_by, 
        ascending=ascending, 
        only_bookmarked=only_bookmarked, 
        timeframe=timeframe
    )

@app.get("/api/teams/{team_id}/roster")
def get_team_roster(
    team_id: int,
    sort_by: str = "fppg",
    ascending: bool = False,
    timeframe: Optional[str] = "2027"
):
    roster_players = fetch_team_roster(LEAGUE_ID, team_id=team_id, espn_s2=ESPN_S2, swid=SWID, is_public=False)
    return filter_and_sort_players(
        roster_players,
        sort_by=sort_by,
        ascending=ascending,
        timeframe=timeframe
    )

@app.get("/api/players/search")
def search_all_players(query: Optional[str] = None):
    all_players = fetch_all_players_search(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)
    if query:
        return [p for p in all_players if query.lower() in p.name.lower()]
    return all_players

@app.get("/api/players/top-performers")
def get_top_performers():
    return fetch_top_today_performers(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)

@app.get("/api/trades/evaluate")
def evaluate_trade_teams(user_team_id: int, target_team_id: int):
    return fetch_trade_evaluator_data(LEAGUE_ID, user_team_id, target_team_id, espn_s2=ESPN_S2, swid=SWID, is_public=False)

@app.post("/api/trades/evaluate")
def grade_trade(request: TradeEvaluationRequest):
    trade_data = fetch_trade_finder_data(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)
    if not trade_data or "error" in trade_data:
        raise HTTPException(status_code=502, detail="Could not load league data for trade evaluation")
    teams = trade_data.get("teams", [])
    user_team = next((team for team in teams if team["id"] == request.user_team_id), None)
    target_team = next((team for team in teams if team["id"] == request.target_team_id), None)
    if user_team is None or target_team is None:
        raise HTTPException(status_code=404, detail="One or both teams were not found")
    trade_data["user_team"] = user_team
    trade_data["target_team"] = target_team
    try:
        return evaluate_trade(
            trade_data,
            request.user_gives,
            request.target_gives,
            request.waiver_player,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

@app.get("/api/trades/find")
def find_trade_suggestions(
    user_team_id: int,
    target_team_id: Optional[int] = None,
    limit: int = Query(default=10, ge=1, le=50),
):
    trade_data = fetch_trade_finder_data(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)
    if not trade_data or "error" in trade_data:
        raise HTTPException(status_code=502, detail="Could not load league data for trade finder")
    try:
        return find_trades(trade_data, user_team_id, target_team_id, limit)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

@app.get("/api/ai/context-payload")
def get_ai_context_payload(user_team_id: int):
    return build_full_ai_context_payload(LEAGUE_ID, user_team_id=user_team_id, espn_s2=ESPN_S2, swid=SWID, is_public=False)

@app.get("/api/ai/trade-advice")
def get_ai_trade_advice(user_team_id: int, target_team_id: int):
    trade_data = fetch_trade_finder_data(LEAGUE_ID, espn_s2=ESPN_S2, swid=SWID, is_public=False)
    if not trade_data or "error" in trade_data:
        raise HTTPException(status_code=502, detail="Could not load team data for trade advice")
    try:
        suggestions = find_trades_between_teams(
            trade_data,
            user_team_id,
            target_team_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if suggestions:
        return suggestions[0]
    return {
        "recommendation_type": "STAND_PAT",
        "trade_grade": "N/A",
        "trade_details": {"user_gives": [], "target_gives": [], "waiver_add": None, "waiver_team": None},
        "strategic_rationale": "No mutually beneficial trade was found from the available roster and projection data.",
    }