"""Encode the protocols of a task with Llama-3.1-8B-Instruct and the SAE, and store the concepts present in each.

    python scripts/encode_protocols.py --task completion --out data/features

Needs a GPU with about 20 GB of memory and access to meta-llama/Llama-3.1-8B-Instruct on the Hugging Face Hub.
The output has the format of artifacts/features/{task}.npz, which holds the result of this script for the five tasks.
"""
import argparse

from _common import add_task, tasks
from sae_audit import artifacts, data
from sae_audit.encode import Encoder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    vocabulary = artifacts.vocabulary()
    concepts = sorted(f for f, e in vocabulary.items() if e["mcc"] >= 0.7)
    encoder = Encoder()
    for task in tasks(args.task):
        d, splits = data.load(task, args.data), {}
        for split in ("train", "test"):
            print(f"[{task}] encoding {len(d[split]):,} {split} protocols", flush=True)
            encoded = encoder.encode([r["text"] for r in d[split]], args.batch_size, progress=True)
            splits[split] = ([r["id"] for r in d[split]], [e[0] for e in encoded], [r["y"] for r in d[split]])
        artifacts.save_features(f"{args.out}/{task}.npz", concepts, splits)
        print(f"[{task}] -> {args.out}/{task}.npz")


if __name__ == "__main__":
    main()
