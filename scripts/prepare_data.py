"""Build the trials of a task from the public benchmark and write data/{task}.json.

    python scripts/prepare_data.py --task all

Each file holds the training and test trials of a task: id, protocol text (cut at 3,000 characters) and outcome.
TrialPanorama and TrialBench are downloaded from the Hugging Face Hub on first use.
"""
import argparse

from _common import add_task, tasks
from sae_audit import data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    for task in tasks(args.task):
        d = data.build(task)
        data.save(d, f"{args.out}/{task}.json")
        rate = sum(r["y"] for r in d["train"]) / len(d["train"])
        print(f"[{task}] train {len(d['train']):,} | test {len(d['test']):,} | positive rate {rate:.3f}"
              + (f" | median {d['median']:g}" if "median" in d else "") + f" -> {args.out}/{task}.json")


if __name__ == "__main__":
    main()
