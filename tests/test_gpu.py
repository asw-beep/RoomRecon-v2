import pytest
from roomrecon.gpu import parse_mig_memory, parse_query_memory

# Real `nvidia-smi` excerpt from the GPU machine, 2026-10-03.
SMI = """\
+-----------------------------------------------------------------------------------------+
| MIG devices:                                                                            |
+------------------+----------------------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |                     Memory-Usage |        Vol|      Shared           |
|      ID  ID  Dev |                       BAR1-Usage | SM     Unc| CE ENC DEC OFA JPG    |
|==================+==================================+===========+=======================|
|  0    0   0   0  |           35278MiB / 40447MiB    | 98      0 |  7   0    5    1    1 |
|                  |                13MiB / 65536MiB  |           |                       |
+------------------+----------------------------------+-----------+-----------------------+
"""


def test_parses_mig_row_not_bar1_row():
    assert parse_mig_memory(SMI) == (35278, 40447)


def test_no_mig_table_raises():
    with pytest.raises(ValueError, match="no MIG device row"):
        parse_mig_memory("| NVIDIA-SMI 555.42.06 |\n")


def test_query_output_of_a_plain_gpu_first_device():
    # `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`, two T4s
    assert parse_query_memory("312, 15360\n0, 15360\n") == (312, 15360)


def test_query_output_rejects_mig_permission_message():
    with pytest.raises(ValueError):
        parse_query_memory("[Insufficient Permissions], [Insufficient Permissions]\n")
