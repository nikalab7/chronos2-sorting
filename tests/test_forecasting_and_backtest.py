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


if __name__ == "__main__":
    unittest.main()
