import json
import numpy as np
from roomrecon.scene_cli import load_c2w


def test_load_c2w_list_of_3x4(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"file_path": "a.png", "transform": np.eye(4)[:3].tolist()}] * 2))
    c2w = load_c2w(p)
    assert c2w.shape == (2, 4, 4) and np.allclose(c2w[0], np.eye(4))


def test_load_c2w_frames_dict_4x4(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"frames": [{"transform_matrix": np.eye(4).tolist()}]}))
    assert load_c2w(p).shape == (1, 4, 4)
