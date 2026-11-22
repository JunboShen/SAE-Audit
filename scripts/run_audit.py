"""Audit an LLM with SAE-Audit on the evaluation protocols of a task and compare it with the LLM alone.

    export SAE_AUDIT_API_BASE=https://api.deepseek.com/v1
    export SAE_AUDIT_API_KEY=...
    python scripts/prepare_data.py --task completion
    python scripts/run_audit.py --task completion --model deepseek-v4-flash

By default the task runs in its setting of Table 1, with three runs per protocol on the 250 evaluation protocols.
Every call is cached, so an interrupted run resumes where it stopped. Needs no GPU: the encoded protocols, the
concept probe and the task summary are read from artifacts/.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _common import ROOT, add_task, tasks
from sae_audit import artifacts, audit, data
from sae_audit.llm import LLM
from sae_audit.metrics import balanced_accuracy, majority, paired_bootstrap
from sae_audit.probe import predict
from sae_audit.tasks import TASKS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--model", required=True, help="model name at the endpoint")
    ap.add_argument("--setting", choices=["draft_first", "evidence_first"], help="default: the setting of Table 1")
    ap.add_argument("--auditor", action=argparse.BooleanOptionalAction, help="default: as in Table 1")
    ap.add_argument("--inverted", action="store_true", help="control: every direction of the evidence is inverted")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="use only the first N evaluation protocols")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--extra-body", default="{}", help="JSON with further fields of the request body")
    args = ap.parse_args()
    for name in tasks(args.task):
        run(TASKS[name], args)


def run(task, args):
    setting = args.setting or task.setting
    auditor = task.auditor if args.auditor is None else args.auditor
    d = data.load(task.name, args.data)
    model = artifacts.evidence_model(task.name, {r["id"]: r["text"] for r in d["train"]})
    summary = artifacts.summary(task.name)
    test = artifacts.features(task.name)["test"]
    features, outcome = dict(zip(test["ids"], test["features"])), dict(zip(test["ids"], test["y"]))
    text = {r["id"]: r["text"] for r in d["test"]}
    ids = artifacts.evaluation_ids(task.name)[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.model}_{task.name}_{setting}{'_auditor' if auditor else ''}{'_inverted' if args.inverted else ''}".replace("/", "-")
    llm = LLM(args.model, cache=os.path.join(args.out, args.model.replace("/", "-") + ".cache.jsonl"),
              extra=json.loads(args.extra_body))
    # calls per run: the unaided answer, then the final call, with the auditor one more, and one to read a rewritten analysis
    per_run = 2 + (1 if auditor else 0) + (1 if auditor and setting == "draft_first" else 0)
    print(f"[{task.name}] {args.model} | {setting}{' with the auditor' if auditor else ''} | {len(ids)} protocols x "
          f"{args.runs} runs, at most {len(ids) * args.runs * per_run:,} calls (cached calls are not repeated)", flush=True)

    def one(rid):
        alone = [audit.parse_answer(task, llm.chat(*audit.unaided_prompt(task, text[rid]),
                                                   key=f"{args.model}|{task.name}|{rid}|unaided|{run}"))
                 for run in range(args.runs)]
        answer, runs = audit.audit(llm, task, model, text[rid], features[rid], summary, setting, auditor, args.runs,
                                   args.inverted, key=rid)
        return alone, [r.get("answer") for r in runs]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, ids))
    y = np.array([outcome[i] for i in ids])
    alone = np.array([majority(a, 0) for a, _ in results])
    audited = np.array([majority(b, alone[k]) for k, (_, b) in enumerate(results)])
    changed = audited != alone
    probe = predict(artifacts.probe(task.name), [features[i] for i in ids])[1]
    print(f"  LLM alone        {100 * balanced_accuracy(alone, y):.1f}")
    print(f"  Concept probe    {100 * balanced_accuracy(probe, y):.1f}")
    print(f"  LLM + SAE-Audit  {100 * balanced_accuracy(audited, y):.1f}   "
          f"({int(changed.sum())} answers changed: {int((changed & (audited == y)).sum())} corrected, "
          f"{int((changed & (audited != y)).sum())} made wrong)")
    if len(set(y.tolist())) == 2 and len(y) >= 50:
        gain, lo, hi = paired_bootstrap(y, alone, audited)
        print(f"  improvement      {100 * gain:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"task": task.name, "llm": args.model, "setting": setting, "auditor": auditor, "inverted": args.inverted,
                   "ids": ids, "y": y.tolist(), "llm_alone": [a for a, _ in results], "sae_audit": [b for _, b in results]}, fh)
    print(f"  answers -> {os.path.join(args.out, name + '.json')} | {llm.calls:,} new calls")


if __name__ == "__main__":
    main()
