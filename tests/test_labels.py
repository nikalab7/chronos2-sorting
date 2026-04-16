from __future__ import annotations

import math
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from chronos_equity_prototype.labels import LabelBuilder


class LabelBuilderTests(unittest.TestCase):
    def test_build_computes_residual_returns_against_spy(self) -> None:
        tz = ZoneInfo("America/New_York")
        timestamp = pd.Timestamp("2026-04-10 09:30", tz=tz)
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "session_date": pd.Timestamp("2026-04-10", tz=tz),
                    "slot_id": 1,
                    "timestamp": timestamp,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 102.0,
                    "volume": 1000,
                },
                {
                    "ticker": "SPY",
                    "session_date": pd.Timestamp("2026-04-10", tz=tz),
                    "slot_id": 1,
                    "timestamp": timestamp,
                    "open": 200.0,
                    "high": 201.0,
                    "low": 199.5,
                    "close": 202.0,
                    "volume": 2000,
                },
            ]
        )

        labeled = LabelBuilder().build(frame, benchmark="SPY")
        aapl = labeled[labeled["ticker"] == "AAPL"].iloc[0]

        expected = math.log(102.0 / 100.0) - math.log(202.0 / 200.0)
        self.assertAlmostEqual(aapl["y_slot"], expected, places=12)
        self.assertAlmostEqual(aapl["spy_same_slot_return"], math.log(202.0 / 200.0), places=12)


if __name__ == "__main__":
    unittest.main()
