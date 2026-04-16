from __future__ import annotations

import unittest

import pandas as pd

from chronos_equity_prototype.pipeline import ZeroShotResearchPipeline


class PipelineRankICTests(unittest.TestCase):
    def test_compute_rank_ic_uses_existing_realized_column_without_merge_suffix_bug(self) -> None:
        pipeline = ZeroShotResearchPipeline()
        forecasts = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "prediction_session": pd.Timestamp("2026-04-11"),
                    "score_q50": 0.8,
                    "realized_day_return": 0.03,
                },
                {
                    "ticker": "MSFT",
                    "prediction_session": pd.Timestamp("2026-04-11"),
                    "score_q50": 0.5,
                    "realized_day_return": 0.01,
                },
            ]
        )
        realized = pd.DataFrame(
            [
                {"ticker": "AAPL", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": 0.03},
                {"ticker": "MSFT", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": 0.01},
            ]
        )

        rank_ic = pipeline.compute_rank_ic(forecasts, realized)

        self.assertIn("rank_ic", rank_ic.columns)
        self.assertFalse(rank_ic["rank_ic"].dropna().empty)

    def test_compute_rank_ic_returns_nan_for_constant_scores_without_crashing(self) -> None:
        pipeline = ZeroShotResearchPipeline()
        forecasts = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "prediction_session": pd.Timestamp("2026-04-11"),
                    "score_q50": 0.5,
                    "realized_day_return": 0.03,
                },
                {
                    "ticker": "MSFT",
                    "prediction_session": pd.Timestamp("2026-04-11"),
                    "score_q50": 0.5,
                    "realized_day_return": 0.01,
                },
            ]
        )
        realized = pd.DataFrame(
            [
                {"ticker": "AAPL", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": 0.03},
                {"ticker": "MSFT", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": 0.01},
            ]
        )

        rank_ic = pipeline.compute_rank_ic(forecasts, realized)
        daily_rows = rank_ic[rank_ic["prediction_session"].notna()]

        self.assertEqual(len(daily_rows), 1)
        self.assertTrue(pd.isna(daily_rows.iloc[0]["rank_ic"]))


if __name__ == "__main__":
    unittest.main()
