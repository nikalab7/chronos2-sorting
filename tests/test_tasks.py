from __future__ import annotations

import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from chronos_equity_prototype.tasks import ChronosTaskBuilder


class ChronosTaskBuilderTests(unittest.TestCase):
    def test_build_tasks_aligns_history_and_future_covariates(self) -> None:
        tz = ZoneInfo("America/New_York")
        rows = []
        session_dates = [pd.Timestamp("2026-04-08", tz=tz), pd.Timestamp("2026-04-09", tz=tz), pd.Timestamp("2026-04-10", tz=tz)]
        for ticker, scale in [("AAPL", 1.0), ("SPY", 0.5)]:
            for session_idx, session_date in enumerate(session_dates):
                for slot_id in range(1, 8):
                    rows.append(
                        {
                            "ticker": ticker,
                            "session_date": session_date,
                            "slot_id": slot_id,
                            "timestamp": session_date + pd.Timedelta(hours=slot_id),
                            "open": 100,
                            "high": 101,
                            "low": 99,
                            "close": 100,
                            "volume": 1000,
                            "slot_log_return": 0.0,
                            "benchmark_slot_return": 0.0,
                            "y_slot": scale * (session_idx + slot_id / 10.0),
                            "log_volume": 1.0,
                            "high_low_range": 0.02,
                            "weekday": float(session_date.weekday()),
                            "month_end_flag": 0.0,
                            "spy_same_slot_return": 0.01,
                            "realized_vol_20": 0.03,
                        }
                    )

        frame = pd.DataFrame(rows)
        tasks = ChronosTaskBuilder(benchmark_ticker="SPY").build_tasks(
            frame,
            context_length=14,
            prediction_length=7,
            use_covariates=True,
        )

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.ticker, "AAPL")
        self.assertEqual(task.target.shape[0], 14)
        self.assertEqual(task.realized_future_path.shape[0], 7)
        self.assertIn("log_volume", task.past_covariates)
        self.assertIn("slot_id", task.future_covariates)
        self.assertEqual(task.future_covariates["slot_id"].shape[0], 7)

    def test_build_latest_tasks_targets_next_trading_session_after_market_holiday(self) -> None:
        tz = ZoneInfo("America/New_York")
        rows = []
        session_dates = [pd.Timestamp("2026-04-01", tz=tz), pd.Timestamp("2026-04-02", tz=tz)]
        for ticker, scale in [("AAPL", 1.0), ("SPY", 0.5)]:
            for session_idx, session_date in enumerate(session_dates):
                for slot_id in range(1, 8):
                    rows.append(
                        {
                            "ticker": ticker,
                            "session_date": session_date,
                            "slot_id": slot_id,
                            "timestamp": session_date + pd.Timedelta(hours=slot_id),
                            "open": 100,
                            "high": 101,
                            "low": 99,
                            "close": 100,
                            "volume": 1000,
                            "slot_log_return": 0.0,
                            "benchmark_slot_return": 0.0,
                            "y_slot": scale * (session_idx + slot_id / 10.0),
                            "log_volume": 1.0,
                            "high_low_range": 0.02,
                            "weekday": float(session_date.weekday()),
                            "month_end_flag": 0.0,
                            "spy_same_slot_return": 0.01,
                            "realized_vol_20": 0.03,
                        }
                    )

        frame = pd.DataFrame(rows)
        tasks = ChronosTaskBuilder(benchmark_ticker="SPY").build_latest_tasks(
            frame,
            context_length=14,
            prediction_length=7,
            use_covariates=True,
        )

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.signal_session, pd.Timestamp("2026-04-02", tz=tz))
        self.assertEqual(task.prediction_session, pd.Timestamp("2026-04-06", tz=tz))
        self.assertTrue(pd.isna(task.realized_day_return))
        self.assertEqual(task.future_covariates["slot_id"].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertTrue((task.future_covariates["weekday"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
