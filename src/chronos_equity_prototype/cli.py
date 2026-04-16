from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import DEFAULT_LIQUID_SP500, ForecasterConfig, ResearchConfig
from .pipeline import ZeroShotResearchPipeline


def _model_slug(model_id: str) -> str:
    slug = model_id.strip().lower().replace("/", "_").replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in slug)


def _parse_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if args.tickers_file:
        content = Path(args.tickers_file).read_text(encoding="utf-8")
        return [line.strip().upper() for line in content.splitlines() if line.strip()]
    return list(DEFAULT_LIQUID_SP500)


def _default_start_end() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=59)
    return start.isoformat(), end.isoformat()


def _default_prediction_window() -> tuple[str, str]:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=59)
    return start.isoformat(), end.isoformat()


def _build_experiment_name(args: argparse.Namespace) -> str:
    model_slug = _model_slug(args.model_id)
    return f"{model_slug}_ctx{args.context_length}_cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the zero-shot Chronos-2 equity prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default experiment or the ablation matrix.")
    run_parser.add_argument("--tickers", help="Comma-separated tickers. Defaults to a liquid S&P 500 sample.")
    run_parser.add_argument("--tickers-file", help="Path to a newline-delimited ticker file.")
    run_parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format.")
    run_parser.add_argument("--end", help="Exclusive end date in YYYY-MM-DD format.")
    run_parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    run_parser.add_argument("--model-id", default="amazon/chronos-2")
    run_parser.add_argument("--prediction-length", type=int, default=7)
    run_parser.add_argument("--context-length", type=int, default=256)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--n-long", type=int, default=5)
    run_parser.add_argument("--n-short", type=int, default=5)
    run_parser.add_argument("--benchmark", default="SPY")
    run_parser.add_argument("--cross-learning", action="store_true", help="Enable Chronos-2 cross-learning.")
    run_parser.add_argument("--no-covariates", action="store_true", help="Run the pure-univariate variant.")
    run_parser.add_argument("--matrix", action="store_true", help="Run the plan ablation matrix.")

    predict_parser = subparsers.add_parser(
        "predict-next",
        help="Forecast the next regular session from the latest complete market session.",
    )
    predict_parser.add_argument("--tickers", help="Comma-separated tickers. Defaults to a liquid S&P 500 sample.")
    predict_parser.add_argument("--tickers-file", help="Path to a newline-delimited ticker file.")
    predict_parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format.")
    predict_parser.add_argument("--end", help="Exclusive end date in YYYY-MM-DD format.")
    predict_parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    predict_parser.add_argument("--model-id", default="amazon/chronos-2")
    predict_parser.add_argument("--prediction-length", type=int, default=7)
    predict_parser.add_argument("--context-length", type=int, default=256)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--n-long", type=int, default=5)
    predict_parser.add_argument("--n-short", type=int, default=5)
    predict_parser.add_argument("--benchmark", default="SPY")
    predict_parser.add_argument("--cross-learning", action="store_true", help="Enable Chronos-2 cross-learning.")
    predict_parser.add_argument("--no-covariates", action="store_true", help="Run the pure-univariate variant.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start, end = _default_start_end() if args.command == "run" else _default_prediction_window()
    if args.start:
        start = args.start
    if args.end:
        end = args.end

    tickers = _parse_tickers(args)
    research_config = ResearchConfig(
        model_id=args.model_id,
        prediction_length=args.prediction_length,
        default_context_length=args.context_length,
        benchmark_ticker=args.benchmark.upper(),
        batch_size=args.batch_size,
        n_long=args.n_long,
        n_short=args.n_short,
        liquid_universe=tuple(tickers),
    )
    pipeline = ZeroShotResearchPipeline(research_config=research_config)
    labeled_frame = pipeline.prepare_labeled_frame(tickers=tickers, start=start, end=end)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)

    if args.command == "run" and args.matrix:
        matrix_results = pipeline.run_matrix(labeled_frame)
        for experiment_name, result in matrix_results.items():
            ZeroShotResearchPipeline.write_result(output_dir / experiment_name, result)
        baseline_results = pipeline.run_baselines(labeled_frame)
        for baseline_name, result in baseline_results.items():
            ZeroShotResearchPipeline.write_result(output_dir / baseline_name, result)
        print(f"Wrote matrix outputs to {output_dir}")
        return

    forecaster_config = ForecasterConfig(
        model_id=args.model_id,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        quantiles=(0.1, 0.5, 0.9),
        cross_learning=args.cross_learning,
        use_covariates=not args.no_covariates,
        batch_size=args.batch_size,
    )
    experiment_name = _build_experiment_name(args)
    if args.command == "predict-next":
        result = pipeline.run_next_session_prediction(labeled_frame, forecaster_config, experiment_name)
        ZeroShotResearchPipeline.write_live_prediction(output_dir / experiment_name, result)
        if result.book.frame.empty:
            print("No next-session prediction was produced. Check history length and universe coverage.")
            return

        summary = result.prediction_summary.iloc[0]
        preview = result.book.frame[["side", "rank", "ticker", "score_q50"]].sort_values(["side", "rank", "ticker"])
        print(
            "Predicted next session book for "
            f"{pd.Timestamp(summary['prediction_session']).date()} using signal session "
            f"{pd.Timestamp(summary['signal_session']).date()}:"
        )
        print(preview.to_string(index=False))
        print(f"Wrote next-session outputs to {output_dir / experiment_name}")
        return

    result = pipeline.run_experiment(labeled_frame, forecaster_config, experiment_name)
    ZeroShotResearchPipeline.write_result(output_dir / experiment_name, result)

    baseline_results = pipeline.run_baselines(labeled_frame)
    for baseline_name, baseline_result in baseline_results.items():
        ZeroShotResearchPipeline.write_result(output_dir / baseline_name, baseline_result)

    print(f"Wrote experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
