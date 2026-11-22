"""Build the concept vocabulary from scratch: encode passages of past protocols, name the SAE features, verify the names.

    python scripts/build_vocabulary.py --name-model MODEL --verify-model MODEL --out data/vocabulary.json

Needs a GPU for the encoding and two LLM calls per candidate feature; 20,000 passages give about 11,000 candidates.
The released vocabulary, artifacts/vocabulary.json, is the output of this procedure.
"""
import argparse
import json
import os

from _common import ROOT  # noqa: F401
from sae_audit import data, vocabulary
from sae_audit.encode import LAYER, MODEL, SAE_REPO, TOP_K, Encoder
from sae_audit.llm import LLM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name-model", required=True)
    ap.add_argument("--verify-model", required=True)
    ap.add_argument("--passages", type=int, default=20000)
    ap.add_argument("--out", default="data/vocabulary.json")
    args = ap.parse_args()
    rows = data._jsonl(data._files(data.TRIALPANORAMA, "trial_completion_assessment/train.jsonl")[0])
    units = vocabulary.passages(rows, args.passages)
    trials, texts = [t for t, _ in units], [s for _, s in units]
    print(f"[vocabulary] encoding {len(texts):,} passages", flush=True)
    encoded = Encoder().encode(texts, progress=True)
    cache = os.path.splitext(args.out)[0]
    concepts = vocabulary.build(trials, texts, encoded, LLM(args.name_model, cache=cache + ".names.jsonl"),
                                LLM(args.verify_model, cache=cache + ".verified.jsonl"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL, "sae": SAE_REPO, "layer": LAYER, "features_per_token": TOP_K, "passages": len(texts),
                   "concepts": concepts}, fh, ensure_ascii=False, indent=0)
    print(f"[vocabulary] -> {args.out}")


if __name__ == "__main__":
    main()
