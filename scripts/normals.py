#!/usr/bin/env python
"""Run DN-Splatter's normals_from_pretrain with a smaller batch.

Its BATCH_SIZE (15) runs out of memory on the shared GPU (~6 GB free) and surfaces as a
PyTorch NVML assert inside the MIG instance. Same model and outputs, fewer images per batch.
Usage (from the DN-Splatter dir): NORMALS_BATCH=4 python normals.py --data-dir <seq> --resolution low
"""
import os

import tyro
from dn_splatter.scripts import normals_from_pretrain as nfp

nfp.BATCH_SIZE = int(os.environ.get("NORMALS_BATCH", "4"))

if __name__ == "__main__":
    tyro.cli(nfp.NormalsFromPretrained).main()
