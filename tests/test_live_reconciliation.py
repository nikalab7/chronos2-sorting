from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chronos_equity_prototype.config import ResearchConfig
from chronos_equity_prototype.pipeline import ZeroShotResearchPipeline


class _FakeYahooProvider:
    def __init__(self) -> None:
        self._market_tz = "America/New_York"

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "60m",
        regular_only: bool = True,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        if interval == "60m":
            return self._make_hourly_bars(tickers)
        if interval == "5m":
            return self._make_execution_bars(tickers)
        raise ValueError(f"Unexpected interval: {interval}")

    def _make_hourly_bars(self, tickers: list[str]) -> pd.DataFrame:
        slot_times = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
        returns = {"AAA": 0.010, "BBB": -0.006, "SPY": 0.002}
        rows: list[dict[str, object]] = []
        for ticker in tickers:
            if ticker not in returns:
                continue
            price = 100.0
            for clock_time in slot_times:
                open_price = price
                close_price = open_price * math.exp(returns[ticker])
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": pd.Timestamp(f"2026-04-16 {clock_time}", tz=self._market_tz),
                        "open": open_price,
                        "high": max(open_price, close_price) * 1.001,
                        "low": min(open_price, close_price) * 0.999,
                        "close": close_price,
                        "volume": 1000.0,
                    }
                )
                price = close_price
        return pd.DataFrame(rows)

    def _make_execution_bars(self, tickers: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        execution_specs = {
            "AAA": {"open": 100.0, "exit": 103.0},
            "BBB": {"open": 100.0, "exit": 98.0},
            "SPY": {"open": 100.0, "exit": 100.8},
        }
        bars = [
            ("09:30", 0.0, 0.3, -0.1, 0.2, 100.0),
            ("09:35", 0.1, 0.35, -0.05, 0.25, 120.0),
            ("09:40", 0.15, 0.4, 0.0, 0.3, 140.0),
        ]
        for ticker in tickers:
            if ticker not in execution_specs:
                continue
            open_price = execution_specs[ticker]["open"]
            exit_price = execution_specs[ticker]["exit"]
            for clock_time, open_shift, high_shift, low_shift, close_shift, volume in bars:
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": pd.Timestamp(f"2026-04-16 {clock_time}", tz=self._market_tz),
                        "open": open_price + open_shift,
                        "high": open_price + high_shift,
                        "low": open_price + low_shift,
                        "close": open_price + close_shift,
                        "volume": volume,
                    }
                )
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": pd.Timestamp("2026-04-16 15:55", tz=self._market_tz),
                    "open": exit_price - 0.2,
                    "high": exit_price + 0.1,
                    "low": exit_price - 0.3,
                    "close": exit_price,
                    "volume": 200.0,
                }
            )
        return pd.DataFrame(rows)


class LiveReconciliationTests(unittest.TestCase):
    def _write_prediction_dir(self, root: Path) -> Path:
        prediction_dir = root / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        forecasts = [
            {
                "ticker": "AAA",
                "basket_id": "growth",
                "signal_session": "2026-04-15T04:00:00+00:00",
                "prediction_session": "2026-04-16T04:00:00+00:00",
                "stock_q10_path": [0.005] * 7,
                "stock_q50_path": [0.010] * 7,
                "stock_q90_path": [0.015] * 7,
                "spy_q10_path": [0.000] * 7,
                "spy_q50_path": [0.002] * 7,
                "spy_q90_path": [0.004] * 7,
                "alpha_q50_path": [0.008] * 7,
                "beta_used": 1.0,
                "score_plain": 0.056,
                "score_risk": 1.0,
                "score_z": 1.0,
                "score_q50": 1.0,
                "beta_mode": "unit_beta_1.0",
            },
            {
                "ticker": "BBB",
                "basket_id": "defensive_value",
                "signal_session": "2026-04-15T04:00:00+00:00",
                "prediction_session": "2026-04-16T04:00:00+00:00",
                "stock_q10_path": [-0.012] * 7,
                "stock_q50_path": [-0.008] * 7,
                "stock_q90_path": [-0.004] * 7,
                "spy_q10_path": [0.000] * 7,
                "spy_q50_path": [0.002] * 7,
                "spy_q90_path": [0.004] * 7,
                "alpha_q50_path": [-0.010] * 7,
                "beta_used": 1.0,
                "score_plain": -0.070,
                "score_risk": -1.0,
                "score_z": -1.0,
                "score_q50": -1.0,
                "beta_mode": "unit_beta_1.0",
            },
        ]
        with (prediction_dir / "forecasts.json").open("w", encoding="utf-8") as handle:
            json.dump(forecasts, handle, indent=2)

        book = pd.DataFrame(
            [
                {
                    "signal_session": "2026-04-15T04:00:00+00:00",
                    "prediction_session": "2026-04-16T04:00:00+00:00",
                    "ticker": "AAA",
                    "side": "long",
                    "rank": 1,
                    "weight": 0.5,
                    "ranking_score_column": "score_z",
                    "ranking_score": 1.0,
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.10,
                    "tail_size": 1,
                    "basket_id": "growth",
                    "score_plain": 0.056,
                    "score_z": 1.0,
                    "score_q50": 1.0,
                    "beta_mode": "unit_beta_1.0",
                },
                {
                    "signal_session": "2026-04-15T04:00:00+00:00",
                    "prediction_session": "2026-04-16T04:00:00+00:00",
                    "ticker": "BBB",
                    "side": "short",
                    "rank": 1,
                    "weight": -0.5,
                    "ranking_score_column": "score_z",
                    "ranking_score": -1.0,
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.10,
                    "tail_size": 1,
                    "basket_id": "defensive_value",
                    "score_plain": -0.070,
                    "score_z": -1.0,
                    "score_q50": -1.0,
                    "beta_mode": "unit_beta_1.0",
                },
            ]
        )
        book.to_csv(prediction_dir / "book.csv", index=False)

        prediction_summary = pd.DataFrame(
            [
                {
                    "experiment_name": "v2_flagship_basket_raw_full_amazon_chronos_2_ctx224_cross0_cov1",
                    "generated_at_utc": "2026-04-15T20:01:00+00:00",
                    "signal_session": "2026-04-15T04:00:00+00:00",
                    "prediction_session": "2026-04-16T04:00:00+00:00",
                    "n_forecasts": 2,
                    "n_longs": 1,
                    "n_shorts": 1,
                    "model_id": "amazon/chronos-2",
                    "context_length": 224,
                    "prediction_length": 7,
                    "cross_learning": False,
                    "use_covariates": True,
                    "basket_mode": "full",
                    "beta_mode": "unit_beta_1.0",
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.10,
                    "tail_size": 1,
                    "score_column": "score_z",
                }
            ]
        )
        prediction_summary.to_csv(prediction_dir / "prediction_summary.csv", index=False)
        return prediction_dir

    def test_reconcile_live_prediction_builds_and_dedupes_live_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_dir = self._write_prediction_dir(root)
            ledger_dir = root / "ledger"

            pipeline = ZeroShotResearchPipeline(
                provider=_FakeYahooProvider(),
                research_config=ResearchConfig(),
            )

            pipeline.reconcile_live_prediction(
                prediction_dir=prediction_dir,
                ledger_dir=ledger_dir,
                execution_interval="5m",
                execution_mode="strict",
            )
            pipeline.reconcile_live_prediction(
                prediction_dir=prediction_dir,
                ledger_dir=ledger_dir,
                execution_interval="5m",
                execution_mode="strict",
            )

            self.assertTrue((ledger_dir / "live_reconciled_forecasts.csv").exists())
            self.assertTrue((ledger_dir / "live_scoreboard_positions.csv").exists())
            self.assertTrue((ledger_dir / "live_scoreboard_daily.csv").exists())
            self.assertTrue((ledger_dir / "live_rank_ic.csv").exists())
            self.assertTrue((ledger_dir / "live_metrics_summary.csv").exists())
            self.assertTrue((ledger_dir / "live_execution_sensitivity_comparison.csv").exists())
            self.assertTrue((ledger_dir / "prediction_summary_history.csv").exists())

            reconciled = pd.read_csv(ledger_dir / "live_reconciled_forecasts.csv")
            scoreboard_positions = pd.read_csv(ledger_dir / "live_scoreboard_positions.csv")
            scoreboard_daily = pd.read_csv(ledger_dir / "live_scoreboard_daily.csv")
            execution_summary = pd.read_csv(ledger_dir / "live_execution_sensitivity_comparison.csv")
            prediction_history = pd.read_csv(ledger_dir / "prediction_summary_history.csv")

            self.assertEqual(len(reconciled), 2)
            self.assertEqual(len(scoreboard_positions), 2)
            self.assertEqual(len(scoreboard_daily), 1)
            self.assertEqual(len(prediction_history), 1)
            self.assertIn("next_day_realized_alpha", scoreboard_positions.columns)
            self.assertIn("score_plain", scoreboard_positions.columns)
            self.assertIn("score_z", scoreboard_positions.columns)
            self.assertIn("execution_mode", execution_summary.columns)
            self.assertIn("net_sharpe_10bps", execution_summary.columns)
            exact_open = execution_summary.loc[execution_summary["execution_mode"] == "exact_open"].iloc[0]
            self.assertAlmostEqual(
                float(exact_open["mean_daily_long_short_spread"]),
                2.0 * float(exact_open["mean_daily_gross_return"]),
            )

    def test_reconcile_live_prediction_blocks_future_realized_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_dir = self._write_prediction_dir(root)
            ledger_dir = root / "ledger"

            pipeline = ZeroShotResearchPipeline(
                provider=_FakeYahooProvider(),
                research_config=ResearchConfig(),
            )

            with self.assertRaisesRegex(ValueError, "still in the future"):
                pipeline.reconcile_live_prediction(
                    prediction_dir=prediction_dir,
                    ledger_dir=ledger_dir,
                    execution_interval="5m",
                    execution_mode="strict",
                    now=pd.Timestamp("2026-04-15 18:00:00", tz="America/New_York"),
                )


if __name__ == "__main__":
    unittest.main()
