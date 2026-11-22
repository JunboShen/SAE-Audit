"""Write the task summary: three sentences about the probe of a task, written once by an LLM.

    python scripts/write_task_summary.py --task completion --model MODEL --out data/task_summaries

One call per task. The released summaries are in artifacts/task_summaries/.
"""
import argparse
import os

from _common import add_task, tasks
from sae_audit import artifacts
from sae_audit.llm import LLM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/task_summaries")
    args = ap.parse_args()
    llm = LLM(args.model)
    os.makedirs(args.out, exist_ok=True)
    for task in tasks(args.task):
        text = " ".join((llm.chat(*artifacts.evidence_model(task).summary_prompt(), max_tokens=180) or "").split())
        with open(os.path.join(args.out, f"{task}.txt"), "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"[{task}] {text}")


if __name__ == "__main__":
    main()
