import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kiln_mvp import (  # noqa: E402
    FORBIDDEN_FEATURE_FIELDS,
    INLET_CHEMISTRY_COLUMNS,
    PROCESS_COLUMNS,
    READ_COLUMNS,
    add_minute_time_anchors,
    assert_feature_contract,
    assert_output_safe,
    build_fcao_event_table,
    build_group_delayed_features,
    build_features,
    build_direction_signal,
    extract_change_events,
    fit_material_cluster_components,
    future_window_all_valid,
    future_window_aggregate,
    aggregate_complete_minutes,
    select_fcao_group_delays,
    time_split_masks,
)


class StageATests(unittest.TestCase):
    def test_minute_bucket_anchors_are_explicit(self) -> None:
        times = pd.to_datetime(
            ["2025-01-01 00:00:05", "2025-01-01 00:00:35", "2025-01-01 00:01:05"]
        )
        raw = pd.DataFrame(
            {column: [1.0, 2.0, 3.0] for column in READ_COLUMNS if column != "time"},
            index=range(len(times)),
        )
        raw["time"] = times
        minute = aggregate_complete_minutes(raw)
        self.assertEqual(minute.index.name, "prediction_time")
        self.assertEqual(minute.index[0], pd.Timestamp("2025-01-01 00:01:00"))
        self.assertEqual(minute.loc[minute.index[0], "bucket_start"], pd.Timestamp("2025-01-01 00:00:00"))
        self.assertEqual(minute.loc[minute.index[0], "bucket_end"], minute.index[0])
        self.assertAlmostEqual(minute.loc[minute.index[0], "二次风温"], 1.5)

    def test_future_window_mean_uses_exact_forward_window(self) -> None:
        index = pd.date_range("2025-01-01", periods=8, freq="min", name="prediction_time")
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

    def test_previous_fcao_is_available_before_prediction_time(self) -> None:
        index = pd.date_range("2025-01-01", periods=130, freq="min", name="prediction_time")
        values = np.ones(len(index), dtype=float)
        values[10:20] = 2.0
        values[80:] = 3.0
        minute = pd.DataFrame({"出窑熟料游离钙": values}, index=index)
        events, meta = build_fcao_event_table(minute, prediction_horizon_minutes=60)
        target_prediction_time = pd.Timestamp("2025-01-01 00:20:00")
        self.assertEqual(
            events.loc[target_prediction_time, "previous_historian_observation_time"],
            pd.Timestamp("2025-01-01 00:10:00"),
        )
        available = events["previous_historian_observation_time"].dropna()
        self.assertTrue((available.index > available).all())
        self.assertGreater(meta["immediate_previous_not_strictly_available_events"], 0)

    def test_different_variable_groups_use_independent_delays(self) -> None:
        index = pd.date_range("2025-01-01", periods=8, freq="min", name="prediction_time")
        features = pd.DataFrame(
            {
                "出磨生料KH": np.arange(8, dtype=float),
                "窑喂料量反馈值": np.arange(8, dtype=float) + 100,
                "篦冷机层压": np.arange(8, dtype=float) + 200,
                "material_cluster": np.arange(8, dtype=float) + 300,
            },
            index=index,
        )
        aligned = build_group_delayed_features(
            features,
            index[-1:],
            {
                "outmill_chemistry": 2,
                "preheater_calciner_kiln": 1,
                "kiln_head_grate_cooler": 3,
            },
        )
        self.assertEqual(aligned.iloc[0]["出磨生料KH"], 5.0)
        self.assertEqual(aligned.iloc[0]["material_cluster"], 305.0)
        self.assertEqual(aligned.iloc[0]["窑喂料量反馈值"], 106.0)
        self.assertEqual(aligned.iloc[0]["篦冷机层压"], 204.0)

    def test_delay_selection_does_not_read_test_rows(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="min", name="prediction_time")
        features = pd.DataFrame(
            {
                "出磨生料KH": np.arange(30, dtype=float),
                "窑喂料量反馈值": np.arange(30, dtype=float),
                "篦冷机层压": np.arange(30, dtype=float),
                "material_cluster": np.zeros(30),
                "material_cluster_distance": np.ones(30),
            },
            index=index,
        )
        events = pd.DataFrame(
            {"fcao": np.arange(30, dtype=float) / 10 + 1.0, "previous_fcao": np.arange(30, dtype=float) / 10},
            index=index,
        )
        cfg = {
            "fcao_delay_candidates_minutes": {
                "outmill_chemistry": [0, 1],
                "preheater_calciner_kiln": [0, 1],
                "kiln_head_grate_cooler": [0, 1],
            }
        }
        _, evidence = select_fcao_group_delays(features, events, index, cfg)
        self.assertEqual(evidence["selection_test_rows"], 0)
        self.assertEqual(evidence["selection_calibration"]["end"], str(index[-1]))

    def test_rolling_cluster_fit_end_is_before_fold_test(self) -> None:
        index = pd.date_range("2025-01-01", periods=12, freq="min", name="prediction_time")
        minute = pd.DataFrame(index=index)
        for column in READ_COLUMNS:
            if column == "time":
                continue
            minute[column] = 1.0
        minute["出磨生料CaO"] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
        minute["bucket_start"] = index - pd.Timedelta(minutes=1)
        minute["bucket_end"] = index
        minute["prediction_time"] = index
        cfg = {
            "random_seed": 1,
            "cluster_k_min": 2,
            "cluster_k_max": 2,
            "feature_windows_minutes": [3],
        }
        _, _, _, fit_events, meta = fit_material_cluster_components(minute, cfg, index[8])
        self.assertTrue((fit_events.index < index[8]).all())
        self.assertEqual(meta["fit_end"], str(index[8]))

    def test_future_window_requires_every_minute_to_be_valid(self) -> None:
        index = pd.date_range("2025-01-01", periods=8, freq="min", name="prediction_time")
        valid = pd.Series(True, index=index)
        valid.iloc[4] = False
        result = future_window_all_valid(valid, 2, 4)
        self.assertFalse(result.loc[index[0]])
        self.assertTrue(result.loc[index[3]])

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
