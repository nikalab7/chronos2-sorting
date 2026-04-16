from __future__ import annotations

import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from chronos_equity_prototype.data import SessionNormalizer


class SessionNormalizerTests(unittest.TestCase):
    def test_normalize_keeps_only_complete_regular_sessions(self) -> None:
        tz = ZoneInfo("America/New_York")
        full_slot_times = [
            pd.Timestamp("2026-04-10 09:30", tz=tz),
            pd.Timestamp("2026-04-10 10:30", tz=tz),
            pd.Timestamp("2026-04-10 11:30", tz=tz),
            pd.Timestamp("2026-04-10 12:30", tz=tz),
            pd.Timestamp("2026-04-10 13:30", tz=tz),
            pd.Timestamp("2026-04-10 14:30", tz=tz),
            pd.Timestamp("2026-04-10 15:30", tz=tz),
        ]
        partial_slot_times = full_slot_times[:-1]

        rows = []
        for timestamp in full_slot_times:
            rows.append({"ticker": "AAPL", "timestamp": timestamp, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10})
            rows.append({"ticker": "SPY", "timestamp": timestamp, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10})
        for timestamp in partial_slot_times:
            rows.append({"ticker": "AAPL", "timestamp": timestamp + pd.Timedelta(days=1), "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10})
            rows.append({"ticker": "SPY", "timestamp": timestamp + pd.Timedelta(days=1), "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10})

        frame = pd.DataFrame(rows)
        normalized = SessionNormalizer().normalize(frame)

        self.assertEqual(normalized["session_date"].nunique(), 1)
        self.assertEqual(normalized.groupby(["ticker", "session_date"])["slot_id"].nunique().min(), 7)
        self.assertEqual(sorted(normalized["slot_id"].unique().tolist()), [1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
