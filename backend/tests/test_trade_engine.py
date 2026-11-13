import unittest

from trade_engine import evaluate_trade, find_trades


class TradeEngineTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "teams": [
                {
                    "id": 1,
                    "name": "North",
                    "position_needs": {"SG": 1},
                    "roster": [
                        {"id": 11, "name": "North Guard", "position": "PG", "fppg": 10},
                        {"id": 12, "name": "North Center", "position": "C", "fppg": 10},
                    ],
                },
                {
                    "id": 2,
                    "name": "South",
                    "position_needs": {"PG": 1},
                    "roster": [
                        {"id": 21, "name": "South Wing", "position": "SG", "fppg": 10},
                    ],
                },
            ],
            "top_free_agents": [{"id": 31, "name": "Waiver Forward", "position": "SF", "fppg": 12}],
        }

    def test_evaluates_uneven_trade_with_waiver_player(self):
        trade_data = {
            **self.data,
            "user_team": self.data["teams"][0],
            "target_team": self.data["teams"][1],
        }

        result = evaluate_trade(trade_data, [11, 12], [21])

        self.assertEqual(result["trade_details"]["waiver_add"], "Waiver Forward")
        self.assertEqual(result["trade_details"]["waiver_team"], "user")
        self.assertEqual(result["recommendation_type"], "TRADE")
        self.assertGreater(result["user_team"]["net_projected_points"], 0)
        self.assertGreater(result["target_team"]["net_projected_points"], 0)
        self.assertTrue(result["target_team"]["pitch"])

    def test_finds_trade_that_resolves_both_team_needs(self):
        suggestions = find_trades(self.data, user_team_id=1, target_team_id=2)

        self.assertTrue(suggestions)
        self.assertEqual(suggestions[0]["trade_details"]["user_gives"][0]["name"], "North Guard")
        self.assertEqual(suggestions[0]["trade_details"]["target_gives"][0]["name"], "South Wing")
        self.assertGreater(suggestions[0]["user_team"]["positional_need_improvement"], 0)
        self.assertGreater(suggestions[0]["target_team"]["positional_need_improvement"], 0)

    def test_rejects_unknown_roster_player(self):
        trade_data = {
            **self.data,
            "user_team": self.data["teams"][0],
            "target_team": self.data["teams"][1],
        }
        with self.assertRaises(ValueError):
            evaluate_trade(trade_data, ["Not on roster"], [21])


if __name__ == "__main__":
    unittest.main()
