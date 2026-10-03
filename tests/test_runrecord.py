import json
from roomrecon import runrecord


def _start(tmp_path):
    p = tmp_path / "run.json"
    runrecord.start(p, run_id="r1", room="vr_room", method="dn-splatter",
                    command="ns-train ...", repo_commit="abc", dn_splatter_commit="def",
                    free_mib=5169, machine="gpu-machine A100 MIG")
    return p


def test_start_writes_running_record(tmp_path):
    p = _start(tmp_path)
    rec = json.loads(p.read_text())
    assert rec["status"] == "running"
    assert rec["free_mib_at_start"] == 5169
    assert rec["started_at"].endswith("Z")


def test_finish_ok_with_vram_samples(tmp_path):
    p = _start(tmp_path)
    csv = tmp_path / "vram.csv"
    csv.write_text("100,35000,40447\n110,39000,40447\n120,37000,40447\n")
    rec = runrecord.finish(p, exit_code=0, vram_csv=csv)
    assert rec["status"] == "ok"
    assert rec["gpu_used_mib_at_start"] == 35000
    assert rec["gpu_used_mib_max"] == 39000
    assert rec["duration_s"] >= 0


def test_finish_failed_without_csv(tmp_path):
    p = _start(tmp_path)
    rec = runrecord.finish(p, exit_code=137, vram_csv=tmp_path / "missing.csv")
    assert rec["status"] == "failed" and rec["exit_code"] == 137
    assert "gpu_used_mib_max" not in rec
