from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_PATH = REPO_ROOT / "tests" / "stubs"
CONFIGS_PATH = REPO_ROOT / "tests" / "configs"


@pytest.mark.parametrize(
    ("alg", "train_config_name"),
    [
        ("lsc", "train_lsc_smoke.json"),
        ("heterocache", "train_heterocache_smoke.json"),
    ],
)
def test_train_and_eval_cli_smoke(alg: str, train_config_name: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(STUBS_PATH) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = f"pytest_{alg}"
    train_config_path = CONFIGS_PATH / train_config_name
    eval_config_path = CONFIGS_PATH / "eval_smoke.json"

    train_cmd = [
        sys.executable,
        "train.py",
        alg,
        "--default-config-path",
        str(train_config_path),
        "--outputs-path",
        str(outputs_dir),
        "--timestamp",
        timestamp,
        "--device",
        "cpu",
        "--max-steps",
        "1",
    ]
    train_result = subprocess.run(
        train_cmd,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = outputs_dir / f"{alg}_{timestamp}"
    checkpoint_path = run_dir / "final_checkpoint_path.pt"
    train_log_path = run_dir / "train.log"

    assert checkpoint_path.exists(), f"missing checkpoint for {alg}: {checkpoint_path}"
    assert train_log_path.exists(), f"missing train.log for {alg}: {train_log_path}"
    assert "Final checkpoint:" in train_result.stdout

    eval_cmd = [
        sys.executable,
        "eval.py",
        alg,
        "--default-config-path",
        str(eval_config_path),
        "--outputs-path",
        str(outputs_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--device",
        "cpu",
    ]
    eval_result = subprocess.run(
        eval_cmd,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    eval_log_path = run_dir / "eval.log"
    assert eval_log_path.exists(), f"missing eval.log for {alg}: {eval_log_path}"
    assert "Evaluation log:" in eval_result.stdout

    train_log = train_log_path.read_text(encoding="utf-8")
    eval_log = eval_log_path.read_text(encoding="utf-8")

    assert "Starting training" in train_log
    assert "Starting evaluation" in eval_log
    assert "FINAL MARKDOWN SUMMARY" in eval_log
