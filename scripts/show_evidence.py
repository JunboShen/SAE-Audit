"""Print the evidence E(x) of a test protocol as the LLM receives it. No GPU and no API call.

    python scripts/show_evidence.py --task completion --id test-10

With data/{task}.json (scripts/prepare_data.py) the evidence also lists the most similar training trials.
"""
import argparse
import os

from _common import ROOT, add_task
from sae_audit import artifacts, audit, data
from sae_audit.tasks import TASKS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--id", help="id of a test protocol; default: the first evaluation protocol")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--auditor", action="store_true", help="the form given to the auditor")
    ap.add_argument("--inverted", action="store_true", help="the control with every direction inverted")
    args = ap.parse_args()
    task = TASKS[args.task]
    texts = None
    if os.path.exists(os.path.join(args.data, f"{task.name}.json")):
        texts = {r["id"]: r["text"] for r in data.load(task.name, args.data)["train"]}
    model = artifacts.evidence_model(task.name, texts)
    test = artifacts.features(task.name)["test"]
    rid = args.id or artifacts.evaluation_ids(task.name)[0]
    features = test["features"][test["ids"].index(rid)]
    print(f"# {task.name}, {rid}: outcome {task.outcomes[int(test['y'][test['ids'].index(rid)])]}\n")
    if args.auditor:
        print(audit.auditor_evidence(model, features, artifacts.summary(task.name), args.inverted))
    else:
        words = task.setting == "evidence_first"
        print(model.text(features, words=words, rich=words, cases=texts is not None, inverted=args.inverted))


if __name__ == "__main__":
    main()
