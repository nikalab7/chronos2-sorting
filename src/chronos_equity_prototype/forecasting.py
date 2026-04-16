from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import ForecasterConfig
from .types import ForecastBatch, StockTask


def _default_pipeline_factory(model_id: str):
    # Some transformers/Chronos dependency paths import TensorFlow and print noisy INFO logs.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "chronos-forecasting is required for zero-shot forecasting. Install it with `pip install chronos-forecasting`."
        ) from exc

    return Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")


@dataclass
class ZeroShotForecaster:
    pipeline_factory: Callable[[str], object] = _default_pipeline_factory
    _pipeline_cache: dict[str, object] = field(default_factory=dict, init=False)

    def _get_pipeline(self, model_id: str):
        if model_id not in self._pipeline_cache:
            self._pipeline_cache[model_id] = self.pipeline_factory(model_id)
        return self._pipeline_cache[model_id]

    def predict(self, tasks: list[StockTask], model_id: str, config: ForecasterConfig) -> ForecastBatch:
        if not tasks:
            return ForecastBatch(frame=pd.DataFrame())

        pipeline = self._get_pipeline(model_id)
        inputs = [task.model_input() for task in tasks]
        quantiles, _ = pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=config.prediction_length,
            quantile_levels=list(config.quantiles),
            batch_size=config.batch_size,
            context_length=config.context_length,
            cross_learning=config.cross_learning,
            limit_prediction_length=False,
        )

        records: list[dict[str, object]] = []
        quantile_index = {level: index for index, level in enumerate(config.quantiles)}
        if 0.5 not in quantile_index:
            raise ValueError("The configured quantiles must include 0.5 so score_q50 can be computed.")

        for task, task_quantiles in zip(tasks, quantiles):
            values = np.asarray(task_quantiles)
            if values.ndim != 3:
                raise ValueError(f"Expected a 3D quantile tensor for task {task.ticker}, found shape {values.shape}.")

            univariate_values = values[0]
            q10_path = univariate_values[:, quantile_index[0.1]].tolist() if 0.1 in quantile_index else None
            q50_path = univariate_values[:, quantile_index[0.5]].tolist()
            q90_path = univariate_values[:, quantile_index[0.9]].tolist() if 0.9 in quantile_index else None

            records.append(
                {
                    "ticker": task.ticker,
                    "signal_session": task.signal_session,
                    "prediction_session": task.prediction_session,
                    "q10_path": q10_path,
                    "q50_path": q50_path,
                    "q90_path": q90_path,
                    "score_q50": float(np.sum(q50_path)),
                    "realized_future_path": task.realized_future_path.tolist(),
                    "realized_day_return": task.realized_day_return,
                    "context_length": config.context_length,
                    "cross_learning": config.cross_learning,
                    "use_covariates": config.use_covariates,
                    "model_id": model_id,
                }
            )

        frame = pd.DataFrame.from_records(records).sort_values(["signal_session", "ticker"]).reset_index(drop=True)
        return ForecastBatch(frame=frame)
