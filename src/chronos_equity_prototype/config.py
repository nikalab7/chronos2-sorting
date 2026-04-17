from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_LIQUID_SP500 = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "JPM",
    "XOM",
    "LLY",
    "AVGO",
    "V",
    "MA",
    "COST",
    "WMT",
    "HD",
    "ABBV",
    "PG",
    "MRK",
    "CRM",
    "UNH",
)

DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_PREDICTION_LENGTH = 7
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_V2_CONTEXT_LENGTH = 224
DEFAULT_CONTEXT_ABLATIONS = (224, 256)
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_CROSS_LEARNING_ABLATIONS = (False, True)
DEFAULT_USE_COVARIATE_ABLATIONS = (True, False)
DEFAULT_COSTS_BPS = (10, 20, 40)
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_MARKET_TIMEZONE = "America/New_York"
DEFAULT_BATCH_SIZE = 8
DEFAULT_V2_BETA_MODE = "unit_beta_1.0"
DEFAULT_V2_BOOK_MODE = "tail_fraction"
DEFAULT_V2_TAIL_FRACTION = 0.10
DEFAULT_SHORT_BORROW_FEE_ANNUAL_BPS = 30.0


@dataclass(frozen=True)
class ForecasterConfig:
    model_id: str = DEFAULT_MODEL_ID
    prediction_length: int = DEFAULT_PREDICTION_LENGTH
    context_length: int = DEFAULT_CONTEXT_LENGTH
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    cross_learning: bool = False
    use_covariates: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    beta_mode: str = DEFAULT_V2_BETA_MODE


@dataclass(frozen=True)
class ResearchConfig:
    model_id: str = DEFAULT_MODEL_ID
    prediction_length: int = DEFAULT_PREDICTION_LENGTH
    default_context_length: int = DEFAULT_CONTEXT_LENGTH
    default_v2_context_length: int = DEFAULT_V2_CONTEXT_LENGTH
    candidate_context_lengths: tuple[int, ...] = DEFAULT_CONTEXT_ABLATIONS
    candidate_cross_learning: tuple[bool, ...] = DEFAULT_CROSS_LEARNING_ABLATIONS
    candidate_use_covariates: tuple[bool, ...] = DEFAULT_USE_COVARIATE_ABLATIONS
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    costs_bps: tuple[int, ...] = DEFAULT_COSTS_BPS
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    market_timezone: str = DEFAULT_MARKET_TIMEZONE
    batch_size: int = DEFAULT_BATCH_SIZE
    n_long: int = 5
    n_short: int = 5
    default_v2_beta_mode: str = DEFAULT_V2_BETA_MODE
    default_v2_book_mode: str = DEFAULT_V2_BOOK_MODE
    default_v2_tail_fraction: float = DEFAULT_V2_TAIL_FRACTION
    default_short_borrow_fee_annual_bps: float = DEFAULT_SHORT_BORROW_FEE_ANNUAL_BPS
    liquid_universe: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LIQUID_SP500)
    realized_vol_window: int = 20

    def default_forecaster_config(self, *, use_covariates: bool = True, cross_learning: bool = False) -> ForecasterConfig:
        return ForecasterConfig(
            model_id=self.model_id,
            prediction_length=self.prediction_length,
            context_length=self.default_context_length,
            quantiles=self.quantiles,
            cross_learning=cross_learning,
            use_covariates=use_covariates,
            batch_size=self.batch_size,
        )
