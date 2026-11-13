import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from market_analyzer import analyze_player_market
from matchup_simulator import simulate_matchup
from projection_model import ProjectionCorrectionModel
from waiver_engine import find_ir_stashes, rank_waiver_players


class ProjectionModelTests(unittest.TestCase):
    def test_trains_on_labeled_game_rows_and_adjusts_projection(self):
        rows = [
            {
                "espn_projected_points": 20,
                "actual_points": 20 + ((index - 10) / 10) * 4,
                "minutes_trend": (index - 10) / 10,
                "usage_trend": (index - 10) / 20,
                "injury_status": "ACTIVE",
                "schedule_density": 0.5,
            }
            for index in range(20)
        ]
        model = ProjectionCorrectionModel().fit(rows)

        result = model.adjust_player(
            {
                "fppg": 20,
                "minutes_trend": 0.2,
                "usage_trend": 0.1,
                "injury_status": "ACTIVE",
                "schedule_density": 0.5,
            }
        )

        self.assertIn("adjusted_projection", result)
        self.assertGreater(result["adjusted_projection"], 20)
        self.assertEqual(result["outlook_tag"], "Trending Up: Minutes +20%")
        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "projection-model.json"
            model.save(model_path)
            restored = ProjectionCorrectionModel().load(model_path)
            self.assertAlmostEqual(
                restored.predict_residual(
                    {
                        "minutes_trend": 0.2,
                        "usage_trend": 0.1,
                        "injury_status": "ACTIVE",
                        "schedule_density": 0.5,
                    }
                ),
                model.predict_residual(
                    {
                        "minutes_trend": 0.2,
                        "usage_trend": 0.1,
                        "injury_status": "ACTIVE",
                        "schedule_density": 0.5,
                    }
                ),
            )

    def test_requires_enough_actual_labeled_rows(self):
        with self.assertRaises(ValueError):
            ProjectionCorrectionModel().fit(
                [{"actual_points": 10, "espn_projected_points": 9}]
            )


class MarketWaiverMatchupTests(unittest.TestCase):
    def test_market_analyzer_identifies_buy_low_and_patience(self):
        result = analyze_player_market(
            {
                "name": "Buy Low",
                "projection_delta_7d": -8,
                "minutes_trend": 0.1,
                "usage_trend": 0.05,
                "percent_owned": 90,
                "draft_pick": 10,
                "injury_status": "ACTIVE",
            }
        )

        self.assertEqual(result["market_tag"], "BUY_LOW")
        self.assertGreaterEqual(result["patience_score"], 70)

    def test_waiver_ranking_and_ir_stash(self):
        ranked = rank_waiver_players(
            [
                {"name": "Streamer", "adjusted_projection": 15, "games_remaining": 3},
                {"name": "Low Value", "adjusted_projection": 10, "games_remaining": 1},
            ],
            droppable_players=[{"name": "Bench", "adjusted_projection": 8, "games_remaining": 2}],
        )
        stash = find_ir_stashes(
            [
                {
                    "name": "Injured Star",
                    "injury_status": "OUT",
                    "season_projection": 40,
                    "percent_owned": 75,
                }
            ],
            has_open_ir_slot=True,
            minimum_season_projection=30,
            minimum_percent_owned=50,
        )

        self.assertEqual(ranked[0]["name"], "Streamer")
        self.assertEqual(ranked[0]["net_weekly_gain"], 29)
        self.assertEqual(stash[0]["name"], "Injured Star")
        self.assertEqual(find_ir_stashes([], has_open_ir_slot=False), [])

    def test_matchup_simulation_uses_reproducible_monte_carlo(self):
        kwargs = {
            "user_players": [{"adjusted_projection": 25, "historical_stddev": 5}],
            "opponent_players": [{"adjusted_projection": 20, "historical_stddev": 5}],
            "iterations": 1000,
            "seed": 12,
        }

        first = simulate_matchup(**kwargs)
        second = simulate_matchup(**kwargs)

        self.assertEqual(first, second)
        self.assertGreater(first["win_probability_percent"], 50)
        self.assertEqual(first["variance_quality"], "historical")


if __name__ == "__main__":
    unittest.main()
