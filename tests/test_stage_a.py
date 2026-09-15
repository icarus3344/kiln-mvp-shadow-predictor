import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kiln_mvp import (  # noqa: E402
    FORBIDDEN_FEATURE_FIELDS,
    PROCESS_COLUMNS,
    assert_feature_contract,
    assert_output_safe,
    build_features,
    build_direction_signal,
    extract_change_events,
    future_window_aggregate,
    time_split_masks,
)


class StageATests(unittest.TestCase):
    def test_future_window_mean_uses_exact_forward_window(self) -> None:
        index = pd.date_range("2025-01-01", periods=8, freq="min")
        series = pd.Series(np.arange(8, dtype=float), index=index)
        result = future_window_aggregate(series, 2, 4, "mean")
        self.assertEqual(result.loc[index[0]], 3.0)
        self.assertEqual(result.loc[index[3]], 6.0)

    def test_causal_features_do_not_change_when_future_changes(self) -> None:
        index = pd.date_range("2025-01-01", periods=16, freq="min")
        data = {column: np.arange(16, dtype=float) for column in PROCESS_COLUMNS}
        minute = pd.DataFrame(data, index=index)
        cluster = pd.Series(np.arange(16) % 2, index=index, dtype=float)
        distance = pd.Series(np.arange(16, dtype=float), index=index)
        original = build_features(minute, cluster, distance, [3])
        changed = minute.copy()
        changed.loc[index[10], "二次风温"] = 99999.0
        changed_features = build_features(changed, cluster, distance, [3])
        pd.testing.assert_frame_equal(original.loc[index[:10]], changed_features.loc[index[:10]])

    def test_time_split_has_disjoint_purged_boundaries(self) -> None:
        index = pd.date_range("2025-01-01", periods=80, freq="min")
        valid = pd.Series(True, index=index)
        train, validation, test, _ = time_split_masks(valid, horizon_minutes=3)
        self.assertFalse(set(index[train]).intersection(index[validation]))
        self.assertFalse(set(index[validation]).intersection(index[test]))
        self.assertLess(index[train].max() + pd.Timedelta(minutes=3), index[validation].min())
        self.assertLess(index[validation].max() + pd.Timedelta(minutes=3), index[test].min())

    def test_fcao_events_are_deduplicated(self) -> None:
        index = pd.date_range("2025-01-01", periods=7, freq="min")
        series = pd.Series([np.nan, 1.0, 1.0, 2.0, 2.0, np.nan, 2.0], index=index)
        events = extract_change_events(series)
        self.assertEqual(list(events.index), [index[1], index[3], index[6]])
        self.assertEqual(list(events), [1.0, 2.0, 2.0])

    def test_forbidden_fields_cannot_enter_features(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="min")
        allowed = pd.DataFrame({column: [1.0, 2.0] for column in PROCESS_COLUMNS}, index=index)
        assert_feature_contract(allowed)
        for forbidden in FORBIDDEN_FEATURE_FIELDS[:4]:
            bad = allowed.copy()
            bad[forbidden] = 1.0
            with self.assertRaises(AssertionError):
                assert_feature_contract(bad)

    def test_raw_parquet_is_not_an_output_target(self) -> None:
        source = Path("/tmp/kiln_mvp_stage_a_test_source.parquet")
        assert_output_safe(source, PROJECT_ROOT / "runs" / "test-output")
        with self.assertRaises(ValueError):
            assert_output_safe(source, source)

    def test_failed_model_has_no_objective_or_actuator_recommendation(self) -> None:
        signal = build_direction_signal(1000.0, 1100.0, 7.0, promoted=False)
        self.assertIsNone(signal["objective_direction"])
        self.assertIsNone(signal["actuator_recommendation"])


if __name__ == "__main__":
    unittest.main()
