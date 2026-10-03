import pytest
from roomrecon.resume import remaining_steps


def test_remaining_counts_from_latest_checkpoint(tmp_path):
    for s in (1000, 2000):
        (tmp_path / f"step-{s:09d}.ckpt").write_bytes(b"")
    # nerfstudio treats --max-num-iterations as *additional* steps after a resume
    assert remaining_steps(tmp_path, 3000) == 1000


def test_no_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint"):
        remaining_steps(tmp_path, 3000)


def test_already_finished_raises(tmp_path):
    (tmp_path / "step-000003000.ckpt").write_bytes(b"")
    with pytest.raises(ValueError, match="already reached"):
        remaining_steps(tmp_path, 3000)
