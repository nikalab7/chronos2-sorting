from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StockTask:
    ticker: str
    signal_session: pd.Timestamp
    prediction_session: pd.Timestamp
    target: np.ndarray
    past_covariates: dict[str, np.ndarray]
    future_covariates: dict[str, np.ndarray]
    realized_future_path: np.ndarray
    realized_day_return: float

    def model_input(self) -> dict[str, object]:
        payload: dict[str, object] = {"target": self.target}
        if self.past_covariates:
            payload["past_covariates"] = self.past_covariates
        if self.future_covariates:
            payload["future_covariates"] = self.future_covariates
        return payload


@dataclass(frozen=True)
class ForecastBatch:
    frame: pd.DataFrame


@dataclass(frozen=True)
class Book:
    frame: pd.DataFrame


@dataclass(frozen=True)
class MetricsReport:
    summary: pd.DataFrame
    daily_returns: pd.DataFrame
    position_returns: pd.DataFrame


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    forecasts: ForecastBatch
    book: Book
    metrics: MetricsReport
    rank_ic_summary: pd.DataFrame
    data_quality_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_quality_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class LivePredictionResult:
    experiment_name: str
    forecasts: ForecastBatch
    book: Book
    prediction_summary: pd.DataFrame
    data_quality_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_quality_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
