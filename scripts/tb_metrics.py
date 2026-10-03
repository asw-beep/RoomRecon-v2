#!/usr/bin/env python
"""Dump every eval scalar from nerfstudio's tensorboard events to CSV (step,tag,value).

The final metric alone hid a training collapse in v1; this gives the whole curve.
Usage: python scripts/tb_metrics.py <train_dir> <out.csv>
"""
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

train_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for ev in sorted(train_dir.rglob("events.out.tfevents.*")):
    acc = EventAccumulator(str(ev.parent))
    acc.Reload()
    for tag in acc.Tags()["scalars"]:
        if "Eval" in tag:
            rows += [(e.step, tag, e.value) for e in acc.Scalars(tag)]
rows = sorted(set(rows))
out.write_text("step,tag,value\n" + "".join(f"{s},{t},{v}\n" for s, t, v in rows))
print(f"{len(rows)} rows, tags: {sorted({t for _, t, _ in rows})}")
