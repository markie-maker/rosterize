import unittest

from data_pipeline import build_player_features, process_espn_payload


class DataPipelineTests(unittest.TestCase):
    def test_builds_rolling_projection_and_role_features(self):
        payload = [
            {
                "id": 1,
                "name": "Rising Guard",
                "position": "PG",
                "timeframe_stats": {"proj": {"fppg": 20}},
                "game_logs": [
                    {
                        "date": "2026-11-10",
                        "actual_fantasy_points": 12,
                        "espn_projected_points": 10,
                        "minutes": 30,
                        "fga": 9,
                    },
                    {
                        "date": "2026-11-05",
                        "actual_fantasy_points": 8,
                        "espn_projected_points": 10,
                        "minutes": 30,
                        "fga": 9,
                    },
                    {
                        "date": "2026-10-20",
                        "actual_fantasy_points": 15,
                        "espn_projected_points": 10,
                        "minutes": 25,
                        "fga": 5,
                    },
                ],
            }
        ]
        free_agents = [
            {"id": 2, "position": "PG", "fppg": 10},
            {"id": 3, "position": "PG", "fppg": 7},
        ]

        [features] = build_player_features(
            payload,
            free_agents=free_agents,
            replacement_rank=1,
            as_of="2026-11-12",
        )

        self.assertEqual(features["projection_delta_7d"], 2)
        self.assertEqual(features["projection_delta_14d"], 0)
        self.assertEqual(features["projection_delta_30d"], 5)
        self.assertAlmostEqual(features["minutes_trend"], 0.2)
        self.assertAlmostEqual(features["usage_trend"], 0.8)
        self.assertEqual(features["positional_replacement_values"], {"PG": 10})
        self.assertEqual(features["positional_scarcity"], 10)

    def test_does_not_invent_deltas_without_game_level_projections(self):
        [features] = build_player_features(
            [
                {
                    "id": 1,
                    "name": "No Game Logs",
                    "fppg": 18,
                    "timeframe_stats": {
                        "7d": {"min": 30, "fga": 12},
                        "30d": {"min": 25, "fga": 10},
                    },
                }
            ],
            as_of="2026-11-12",
        )

        self.assertIsNone(features["projection_delta_7d"])
        self.assertIsNone(features["projection_delta_14d"])
        self.assertIsNone(features["projection_delta_30d"])
        self.assertAlmostEqual(features["minutes_trend"], 0.2)
        self.assertAlmostEqual(features["usage_trend"], 0.2)

    def test_accepts_dataframe_and_json_wrappers(self):
        import pandas as pd

        frame = pd.DataFrame([{"id": 1, "name": "DataFrame Player", "fppg": 12}])
        self.assertEqual(process_espn_payload(frame)[0]["name"], "DataFrame Player")
        self.assertEqual(process_espn_payload({"players": [{"id": 2, "name": "JSON Player"}]})[0]["id"], 2)


if __name__ == "__main__":
    unittest.main()
