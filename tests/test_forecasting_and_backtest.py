from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chronos_equity_prototype.backtest import BacktestRunner
from chronos_equity_prototype.config import DEFAULT_MODEL_ID, ForecasterConfig
from chronos_equity_prototype.forecasting import ZeroShotForecaster
from chronos_equity_prototype.portfolio import PortfolioBuilder
from chronos_equity_prototype.types import StockTask


class _FakePipeline:
    def predict_quantiles(
        self,
        *,
        inputs,
        prediction_length,
        quantile_levels,
        batch_size,
        context_length,
        cross_learning,
        limit_prediction_length,
    ):
        del inputs, batch_size, context_length, cross_learning, limit_prediction_length
        outputs = []
        for _ in range(2):
            tensor = np.zeros((1, prediction_length, len(quantile_levels)), dtype=float)
            for horizon in range(prediction_length):
                tensor[0, horizon, 0] = -0.01
                tensor[0, horizon, 1] = 0.02
                tensor[0, horizon, 2] = 0.05
            outputs.append(tensor)
        return outputs, outputs

    def predict_df(
        self,
        *,
        df,
        future_df,
        id_column,
        timestamp_column,
        target,
        prediction_length,
        quantile_levels,
        batch_size,
        context_length,
        cross_learning,
        validate_inputs,
    ):
        del future_df, batch_size, context_length, cross_learning, validate_inputs
        item_id = df[id_column].iloc[0]
        start_ts = pd.Timestamp(df[timestamp_column].iloc[-1]) + pd.Timedelta(hours=1)
        rows = []
        for target_name in target:
            for horizon in range(prediction_length):
                rows.append(
                    {
                        id_column: item_id,
                        timestamp_column: start_ts + pd.Timedelta(hours=horizon),
                        "target_name": target_name,
                        "predictions": 0.01,
                        str(quantile_levels[0]): -0.02,
                        str(quantile_levels[1]): 0.01,
                        str(quantile_levels[2]): 0.03,
                    }
                )
        return pd.DataFrame(rows)


class ForecastingAndBacktestTests(unittest.TestCase):
    def test_zero_shot_forecaster_sums_q50_path(self) -> None:
        tasks = [
            StockTask(
                ticker="AAPL",
                signal_session=pd.Timestamp("2026-04-10"),
                prediction_session=pd.Timestamp("2026-04-11"),
                target=np.array([0.1, 0.2]),
                past_covariates={},
                future_covariates={},
                realized_future_path=np.array([0.0] * 7),
                realized_day_return=0.01,
            ),
            StockTask(
                ticker="MSFT",
                signal_session=pd.Timestamp("2026-04-10"),
                prediction_session=pd.Timestamp("2026-04-11"),
                target=np.array([0.1, 0.2]),
                past_covariates={},
                future_covariates={},
                realized_future_path=np.array([0.0] * 7),
                realized_day_return=0.02,
            ),
        ]

        forecaster = ZeroShotForecaster(pipeline_factory=lambda model_id: _FakePipeline())
        forecasts = forecaster.predict(tasks, DEFAULT_MODEL_ID, ForecasterConfig())

        self.assertEqual(len(forecasts.frame), 2)
        self.assertAlmostEqual(forecasts.frame.iloc[0]["score_q50"], 7 * 0.02)

    def test_backtest_runner_reports_gross_and_net(self) -> None:
        forecasts = pd.DataFrame(
            [
                {"ticker": "AAPL", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_q50": 0.8},
                {"ticker": "MSFT", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_q50": 0.6},
                {"ticker": "XOM", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_q50": -0.4},
                {"ticker": "JPM", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_q50": -0.7},
            ]
        )
        book = PortfolioBuilder().rank_and_size(forecasts, n_long=1, n_short=1)
        realized = pd.DataFrame(
            [
                {"ticker": "AAPL", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": 0.03},
                {"ticker": "MSFT", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": 0.02},
                {"ticker": "XOM", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": -0.01},
                {"ticker": "JPM", "prediction_session": pd.Timestamp("2026-04-11"), "realized_day_return": -0.02},
            ]
        )

        report = BacktestRunner().run(book, realized, costs_bps=(10, 20, 40))

        self.assertEqual(set(report.summary["cost_bps"].tolist()), {0, 10, 20, 40})
        self.assertIn("net_return_10bps", report.daily_returns.columns)
        self.assertEqual(len(report.position_returns), 2)

    def test_portfolio_builder_supports_tail_fraction_books(self) -> None:
        forecasts = pd.DataFrame(
            [
                {"ticker": "AAPL", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.8},
                {"ticker": "MSFT", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.6},
                {"ticker": "NVDA", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.2},
                {"ticker": "JPM", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.3},
                {"ticker": "XOM", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.5},
                {"ticker": "PG", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.7},
                {"ticker": "CRM", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.1},
                {"ticker": "COST", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.1},
                {"ticker": "ABBV", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.05},
                {"ticker": "WMT", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.05},
                {"ticker": "AMZN", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.4},
                {"ticker": "META", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.4},
                {"ticker": "GOOGL", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.3},
                {"ticker": "HD", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.2},
                {"ticker": "LLY", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.6},
                {"ticker": "MA", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.25},
                {"ticker": "AVGO", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.45},
                {"ticker": "MRK", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.45},
                {"ticker": "UNH", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": -0.35},
                {"ticker": "C", "signal_session": pd.Timestamp("2026-04-10"), "prediction_session": pd.Timestamp("2026-04-11"), "score_z": 0.15},
            ]
        )

        book = PortfolioBuilder().rank_and_size(
            forecasts,
            n_long=5,
            n_short=5,
            score_column="score_z",
            book_mode="tail_fraction",
            tail_fraction=0.10,
        )

        self.assertEqual(int((book.frame["side"] == "long").sum()), 2)
        self.assertEqual(int((book.frame["side"] == "short").sum()), 2)
        self.assertTrue((book.frame["book_mode"] == "tail_fraction").all())
        self.assertTrue((book.frame["tail_size"] == 2).all())

    def test_predict_dataframe_uses_multivariate_predict_df_interface(self) -> None:
        forecaster = ZeroShotForecaster(pipeline_factory=lambda model_id: _FakePipeline())
        history_df = pd.DataFrame(
            {
                "item_id": ["growth|2026-04-10", "growth|2026-04-10"],
                "timestamp": [pd.Timestamp("2026-04-10 00:00:00"), pd.Timestamp("2026-04-10 01:00:00")],
                "AAPL": [0.01, 0.02],
                "SPY": [0.00, 0.01],
            }
        )

        predictions = forecaster.predict_dataframe(
            history_df,
            model_id=DEFAULT_MODEL_ID,
            config=ForecasterConfig(),
            target_columns=["AAPL", "SPY"],
            future_df=None,
        )

        self.assertEqual(set(predictions["target_name"].tolist()), {"AAPL", "SPY"})
        self.assertEqual(len(predictions), 14)


if __name__ == "__main__":
    unittest.main()
