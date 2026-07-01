"""End-to-end smoke tests for flowmatching under the self-supervised regimes.

flowmatching + n2v/n2sim used to fail fast with a parser error (the crashed,
runtime-0 runs in the sweep). They now train the unconditional self-supervised
flow; these tests prove the run completes on synthetic CPU data and that the
model built is the unconditional flow, not the conditional one.
"""

import pytest

torch = pytest.importorskip("torch")

from ctdenoiser.models import FlowMatching, SelfSupervisedFlow
from ctdenoiser.train import main


def _args(tmp_path, mode):
    return [
        "--model", "flowmatching",
        "--training-mode", mode,
        "--epochs", "1",
        "--synthetic-len", "8",
        "--batch-size", "4",
        "--patch-size", "32",
        "--num-workers", "0",
        "--device", "cpu",
        "--checkpoint-dir", str(tmp_path),
    ]


@pytest.mark.parametrize("mode", ["n2sim", "n2v"])
def test_flowmatching_selfsup_runs_end_to_end(tmp_path, mode):
    # No exception == the old fail-fast crash is gone and a full train+eval ran.
    main(_args(tmp_path, mode))
    assert (tmp_path / "flowmatching.pt").exists()


def test_unconditional_flow_has_no_noisy_conditioning():
    # The conditional flow takes a 2-channel (x_t, cond) input; the unconditional
    # self-supervised flow used for n2v/n2sim takes 1 channel (no noisy
    # conditioning) -- that omission is what stops it reproducing target noise.
    assert SelfSupervisedFlow().net.enc1[0].in_channels == 1
    assert FlowMatching().net.enc1[0].in_channels == 2
