"""SAE-Audit on a new protocol: encode it, build the evidence and audit the answer of an LLM.

    python scripts/audit_protocol.py --task completion --protocol my_protocol.txt --model deepseek-v4-flash

The protocol is encoded locally with Llama-3.1-8B-Instruct and the SAE (GPU); only the audited LLM is called
through the endpoint. With --evidence-only the evidence is printed and no LLM is called.
"""
import argparse
import os

from _common import ROOT, add_task
from sae_audit import artifacts, audit, data
from sae_audit.encode import Encoder
from sae_audit.llm import LLM
from sae_audit.probe import predict
from sae_audit.tasks import TASKS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--protocol", required=True, help="text file with the protocol")
    ap.add_argument("--model", help="audited LLM at the endpoint")
    ap.add_argument("--evidence-only", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    task = TASKS[args.task]
    with open(args.protocol, encoding="utf-8") as fh:
        protocol = fh.read()[:data.MAX_CHARS]
    texts = None
    if os.path.exists(os.path.join(args.data, f"{task.name}.json")):
        texts = {r["id"]: r["text"] for r in data.load(task.name, args.data)["train"]}
    model = artifacts.evidence_model(task.name, texts)
    features = Encoder().features(protocol)
    probability = predict(artifacts.probe(task.name), [features])[0][0]
    words = task.setting == "evidence_first"
    print(model.text(features, words=words, rich=words, cases=texts is not None))
    print(f"Concept probe: probability of '{task.positive}' {probability:.2f}\n")
    if args.evidence_only or not args.model:
        return
    answer, runs = audit.audit(LLM(args.model), task, model, protocol, features, artifacts.summary(task.name),
                               task.setting, task.auditor, args.runs)
    print(runs[0]["final"])
    print(f"\nAnswer of {args.model} with SAE-Audit ({args.runs} runs): "
          f"{task.outcomes[answer] if answer is not None else 'no answer'}")


if __name__ == "__main__":
    main()
