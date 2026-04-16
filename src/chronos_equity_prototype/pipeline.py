from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtest import BacktestRunner
from .baselines import BaselineScorer
from .config import ForecasterConfig, ResearchConfig
from .data import SessionNormalizer, YahooIntradayProvider
from .forecasting import ZeroShotForecaster
from .labels import LabelBuilder
from .portfolio import PortfolioBuilder
from .tasks import ChronosTaskBuilder
from .types import ExperimentResult, ForecastBatch, LivePredictionResult, StockTask


def _model_slug(model_id: str) -> str:
    slug = model_id.strip().lower().replace("/", "_").replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in slug)


@dataclass
class ZeroShotResearchPipeline:
    provider: YahooIntradayProvider | None = None
    normalizer: SessionNormalizer | None = None
    label_builder: LabelBuilder | None = None
    task_builder: ChronosTaskBuilder | None = None
    forecaster: ZeroShotForecaster | None = None
    portfolio_builder: PortfolioBuilder | None = None
    backtest_runner: BacktestRunner | None = None
    baseline_scorer: BaselineScorer | None = None
    research_config: ResearchConfig = ResearchConfig()

    def __post_init__(self) -> None:
        self.provider = self.provider or YahooIntradayProvider()
        self.normalizer = self.normalizer or SessionNormalizer(market_timezone=self.research_config.market_timezone)
        self.label_builder = self.label_builder or LabelBuilder(realized_vol_window=self.research_config.realized_vol_window)
        self.task_builder = self.task_builder or ChronosTaskBuilder(benchmark_ticker=self.research_config.benchmark_ticker)
        self.forecaster = self.forecaster or ZeroShotForecaster()
        self.portfolio_builder = self.portfolio_builder or PortfolioBuilder()
        self.backtest_runner = self.backtest_runner or BacktestRunner()
        self.baseline_scorer = self.baseline_scorer or BaselineScorer(benchmark_ticker=self.research_config.benchmark_ticker)

    def prepare_labeled_frame(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "60m",
    ) -> pd.DataFrame:
        symbols = list(dict.fromkeys([*tickers, self.research_config.benchmark_ticker]))
        raw = self.provider.fetch(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            regular_only=True,
            auto_adjust=True,
        )
        normalized = self.normalizer.normalize(raw)
        return self.label_builder.build(normalized, benchmark=self.research_config.benchmark_ticker)

    def _predict_by_signal_session(self, tasks: list[StockTask], config: ForecasterConfig) -> ForecastBatch:
        if not tasks:
            return ForecastBatch(frame=pd.DataFrame())

        frames: list[pd.DataFrame] = []
        grouped = self.task_builder.group_by_signal_session(tasks)
        for signal_session in sorted(grouped.keys()):
            day_tasks = sorted(grouped[signal_session], key=lambda task: task.ticker)
            for batch_start in range(0, len(day_tasks), config.batch_size):
                batch = day_tasks[batch_start : batch_start + config.batch_size]
                forecasts = self.forecaster.predict(batch, model_id=config.model_id, config=config)
                frames.append(forecasts.frame)

        if not frames:
            return ForecastBatch(frame=pd.DataFrame())
        combined = pd.concat(frames, ignore_index=True).sort_values(["signal_session", "ticker"]).reset_index(drop=True)
        return ForecastBatch(frame=combined)

    def compute_rank_ic(self, forecasts: pd.DataFrame, realized_returns: pd.DataFrame) -> pd.DataFrame:
        def _safe_stat(series: pd.Series, stat: str) -> float:
            clean = series.dropna()
            if clean.empty:
                return float("nan")
            if stat == "mean":
                return float(clean.mean())
            if stat == "median":
                return float(clean.median())
            raise ValueError(f"Unsupported stat: {stat}")

        if "realized_day_return" in forecasts.columns:
            merged = forecasts.copy()
        else:
            merged = forecasts.merge(
                realized_returns[["ticker", "prediction_session", "realized_day_return"]],
                on=["ticker", "prediction_session"],
                how="left",
                validate="many_to_one",
            )
        if merged.empty:
            return pd.DataFrame(columns=["prediction_session", "rank_ic"])
        if merged["realized_day_return"].isna().any():
            raise ValueError("Missing realized_day_return values while computing Rank IC.")

        daily_ic_records: list[dict[str, object]] = []
        for prediction_session, day_frame in merged.groupby("prediction_session", sort=True):
            ranked_scores = day_frame["score_q50"].rank(method="average")
            ranked_returns = day_frame["realized_day_return"].rank(method="average")
            if ranked_scores.nunique(dropna=True) <= 1 or ranked_returns.nunique(dropna=True) <= 1:
                rank_ic = float("nan")
            else:
                rank_ic = float(ranked_scores.corr(ranked_returns))
            daily_ic_records.append({"prediction_session": prediction_session, "rank_ic": rank_ic})

        daily_ic = pd.DataFrame.from_records(daily_ic_records)
        summary = pd.DataFrame(
            [
                {"metric": "mean_rank_ic", "value": _safe_stat(daily_ic["rank_ic"], "mean")},
                {"metric": "median_rank_ic", "value": _safe_stat(daily_ic["rank_ic"], "median")},
                {"metric": "positive_rank_ic_rate", "value": float((daily_ic["rank_ic"] > 0).mean())},
            ]
        )
        return pd.concat([daily_ic, summary], ignore_index=True, sort=False)

    def run_experiment(
        self,
        labeled_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
    ) -> ExperimentResult:
        tasks = self.task_builder.build_tasks(
            labeled_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            use_covariates=forecaster_config.use_covariates,
        )
        forecasts = self._predict_by_signal_session(tasks, forecaster_config)
        realized = self.baseline_scorer.build_realized_daily_returns(labeled_frame)
        rank_ic_summary = self.compute_rank_ic(forecasts.frame, realized)
        book = self.portfolio_builder.rank_and_size(
            forecasts,
            n_long=self.research_config.n_long,
            n_short=self.research_config.n_short,
        )
        metrics = self.backtest_runner.run(book, realized, self.research_config.costs_bps)
        return ExperimentResult(
            experiment_name=experiment_name,
            forecasts=forecasts,
            book=book,
            metrics=metrics,
            rank_ic_summary=rank_ic_summary,
        )

    def run_next_session_prediction(
        self,
        labeled_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
    ) -> LivePredictionResult:
        tasks = self.task_builder.build_latest_tasks(
            labeled_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            use_covariates=forecaster_config.use_covariates,
        )
        forecasts = self.forecaster.predict(tasks, forecaster_config.model_id, forecaster_config)
        book = self.portfolio_builder.rank_and_size(
            forecasts,
            n_long=self.research_config.n_long,
            n_short=self.research_config.n_short,
        )

        if forecasts.frame.empty:
            prediction_summary = pd.DataFrame(
                columns=[
                    "experiment_name",
                    "generated_at_utc",
                    "signal_session",
                    "prediction_session",
                    "n_forecasts",
                    "n_longs",
                    "n_shorts",
                    "model_id",
                    "context_length",
                    "prediction_length",
                    "cross_learning",
                    "use_covariates",
                ]
            )
        else:
            prediction_summary = pd.DataFrame(
                [
                    {
                        "experiment_name": experiment_name,
                        "generated_at_utc": pd.Timestamp.utcnow(),
                        "signal_session": forecasts.frame["signal_session"].iloc[0],
                        "prediction_session": forecasts.frame["prediction_session"].iloc[0],
                        "n_forecasts": int(len(forecasts.frame)),
                        "n_longs": self.research_config.n_long,
                        "n_shorts": self.research_config.n_short,
                        "model_id": forecaster_config.model_id,
                        "context_length": forecaster_config.context_length,
                        "prediction_length": forecaster_config.prediction_length,
                        "cross_learning": forecaster_config.cross_learning,
                        "use_covariates": forecaster_config.use_covariates,
                    }
                ]
            )

        return LivePredictionResult(
            experiment_name=experiment_name,
            forecasts=forecasts,
            book=book,
            prediction_summary=prediction_summary,
        )

    def run_baselines(self, labeled_frame: pd.DataFrame) -> dict[str, ExperimentResult]:
        realized = self.baseline_scorer.build_realized_daily_returns(labeled_frame)
        baseline_scores = self.baseline_scorer.score(labeled_frame)
        results: dict[str, ExperimentResult] = {}
        for baseline_name, baseline_frame in baseline_scores.groupby("baseline_name", sort=True):
            forecast_batch = ForecastBatch(frame=baseline_frame.reset_index(drop=True))
            rank_ic_summary = self.compute_rank_ic(forecast_batch.frame, realized)
            book = self.portfolio_builder.rank_and_size(
                forecast_batch,
                n_long=self.research_config.n_long,
                n_short=self.research_config.n_short,
            )
            metrics = self.backtest_runner.run(book, realized, self.research_config.costs_bps)
            results[baseline_name] = ExperimentResult(
                experiment_name=baseline_name,
                forecasts=forecast_batch,
                book=book,
                metrics=metrics,
                rank_ic_summary=rank_ic_summary,
            )
        return results

    def run_matrix(self, labeled_frame: pd.DataFrame) -> dict[str, ExperimentResult]:
        results: dict[str, ExperimentResult] = {}
        model_slug = _model_slug(self.research_config.model_id)
        for context_length, cross_learning, use_covariates in product(
            self.research_config.candidate_context_lengths,
            self.research_config.candidate_cross_learning,
            self.research_config.candidate_use_covariates,
        ):
            config = ForecasterConfig(
                model_id=self.research_config.model_id,
                prediction_length=self.research_config.prediction_length,
                context_length=context_length,
                quantiles=self.research_config.quantiles,
                cross_learning=cross_learning,
                use_covariates=use_covariates,
                batch_size=self.research_config.batch_size,
            )
            experiment_name = f"{model_slug}_ctx{context_length}_cross{int(cross_learning)}_cov{int(use_covariates)}"
            results[experiment_name] = self.run_experiment(labeled_frame, config, experiment_name)
        return results

    @staticmethod
    def write_result(output_dir: Path, result: ExperimentResult) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.forecasts.frame.to_json(output_dir / "forecasts.json", orient="records", date_format="iso", indent=2)
        result.book.frame.to_csv(output_dir / "book.csv", index=False)
        result.metrics.summary.to_csv(output_dir / "metrics_summary.csv", index=False)
        result.metrics.daily_returns.to_csv(output_dir / "daily_returns.csv", index=False)
        result.rank_ic_summary.to_csv(output_dir / "rank_ic.csv", index=False)

    @staticmethod
    def write_live_prediction(output_dir: Path, result: LivePredictionResult) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.forecasts.frame.to_json(output_dir / "forecasts.json", orient="records", date_format="iso", indent=2)
        result.book.frame.to_csv(output_dir / "book.csv", index=False)
        result.prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
