"""Fit the concept probe of a task on its training trials and report its accuracy.

    python scripts/fit_probe.py --task all

Reads the encoded protocols of artifacts/features (or --features) and needs no GPU and no API call.
"""
import argparse
import json
import os

import numpy as np

from _common import add_task, tasks
from sae_audit import artifacts, probe
from sae_audit.metrics import balanced_accuracy
from sae_audit.tasks import TASKS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--features", help="directory with {task}.npz written by encode_protocols.py")
    ap.add_argument("--out", help="directory in which to write {task}.json")
    args = ap.parse_args()
    vocabulary = artifacts.vocabulary()
    for task in tasks(args.task):
        f = artifacts.features(task, os.path.join(args.features, f"{task}.npz") if args.features else None)
        fitted = probe.fit(TASKS[task], f["train"]["features"], f["train"]["y"], vocabulary)
        y = f["test"]["y"]
        evaluation = set(artifacts.evaluation_ids(task))
        subset = np.array([i in evaluation for i in f["test"]["ids"]])
        label = probe.predict(fitted, f["test"]["features"])[1]
        print(f"[{task}] {len(fitted['concepts'])} concepts | balanced accuracy {100 * balanced_accuracy(label, y):.1f} on "
              f"{len(y):,} test protocols, {100 * balanced_accuracy(label[subset], y[subset]):.1f} on the {int(subset.sum())} "
              f"evaluation protocols")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"{task}.json"), "w", encoding="utf-8") as fh:
                json.dump(fitted, fh, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
