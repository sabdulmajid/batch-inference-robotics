"""Lightweight checks for the bounded CPU recipe; real IQL evidence is separate."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cpu_bounds_reject_unbounded_threads_and_steps():
    smoke = module("rl_cpu_smoke")
    assert smoke.bounded_int("2", minimum=1, maximum=2) == 2
    for value in ("0", "3", "10000"):
        with pytest.raises(argparse.ArgumentTypeError):
            smoke.bounded_int(value, minimum=1, maximum=2)


def test_results_are_not_overwritten(tmp_path):
    smoke = module("rl_cpu_smoke")
    receipt = tmp_path / "result.json"
    receipt.write_text('{"status":"passed"}\n')
    with pytest.raises(FileExistsError):
        smoke.run(argparse.Namespace(output=tmp_path))
    assert receipt.read_text() == '{"status":"passed"}\n'


def test_nested_optimizer_state_comparison_keeps_structure():
    smoke = module("rl_cpu_smoke")
    assert smoke.state_equal({"steps": [1, 2]}, {"steps": [1, 2]})
    assert not smoke.state_equal({"steps": [1, 2]}, {"steps": [1, 3]})
    assert not smoke.state_equal({"steps": [1, 2]}, {"steps": (1, 2)})
    assert not smoke.state_equal({"steps": [1, 2]}, {"other": [1, 2]})


def test_replay_rejects_wrong_source_before_any_simulation(tmp_path):
    prepare = module("prepare_rollout_example")
    source = tmp_path / "data.hdf5"
    source.write_bytes(b"not the pinned real dataset")
    with pytest.raises(ValueError, match="source revision"):
        prepare.fetch_dataset(source)


def test_replay_clock_checks_actual_recorded_times():
    pytest.importorskip("numpy")
    prepare = module("prepare_rollout_example")
    prepare.validate_clock([10.0, 10.05, 10.1], 20)
    for bad in ([0, 0.05, 0.2], [0, 0.05, float("nan")], [0, 0.05, 0.04]):
        with pytest.raises(ValueError):
            prepare.validate_clock(bad, 20)


def test_replay_never_selects_accelerated_macos_graphics(tmp_path, monkeypatch):
    prepare = module("prepare_rollout_example")
    monkeypatch.setattr(prepare.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="OSMesa"):
        prepare.render_example(tmp_path / "unused.hdf5", tmp_path / "render")
    assert not (tmp_path / "render").exists()
