import pytest
from roomrecon.gpu import parse_mig_memory

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
