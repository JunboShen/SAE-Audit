"""Protocols, outcomes and splits of the five tasks, built from the public benchmarks.

TrialPanorama (trial completion, sample size) and TrialBench (adverse events, duration, dropout) are read from the
Hugging Face Hub. Every method reads the same text of a protocol, cut at 3,000 characters. A task keeps at most
6,000 training trials and 1,000 test trials, drawn at random with a fixed seed; the audit is evaluated on 250 of the
test protocols, also fixed.
"""
import glob
import json
import os

import numpy as np

from .tasks import TASKS

TRIALPANORAMA = "TrialPanorama/TrialPanorama-benchmark"
TRIALBENCH = "araag2/TrialBench"
MAX_CHARS = 3000
N_TRAIN, N_TEST, SPLIT_SEED = 6000, 1000, 0
N_EVAL, EVAL_SEED = 250, 3
DESIGN_FIELDS = ("allocation", "intervention_model", "observational_model", "primary_purpose", "time_perspective",
                 "masking", "phase", "number_of_arms")
TRIALBENCH_FIELDS = ("Title", "Summary", "Condition", "Phase", "Eligibility_Criteria", "Interventions", "Study Design",
                     "Additional_Info")


def _files(repo, pattern):
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern])
    return sorted(glob.glob(os.path.join(root, pattern)))


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def protocol_text(row):
    """A TrialPanorama protocol as registered before the trial started: title, design, eligibility criteria, arms."""
    inp = row.get("inputs") or {}
    parts = [f"Title: {inp.get('title') or ''}",
             "Design: " + "; ".join(f"{k}={inp.get(k)}" for k in DESIGN_FIELDS if inp.get(k) is not None)]
    for c in row.get("context") or []:
        if c.get("type") == "eligibility_criteria":
            parts.append(f"Eligibility (gender={c.get('gender')}, age {c.get('minimum_age')}-{c.get('maximum_age')}, "
                         f"healthy_volunteers={c.get('healthy_volunteers')}):\n"
                         + str(c.get("criteria") or "").replace("~", "\n"))
        elif c.get("type") == "arm_design":
            parts.append("Arms:\n" + str(c.get("arms") or ""))
    return "\n\n".join(parts)[:6000]


def _trialbench_text(row):
    parts = []
    for field in TRIALBENCH_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        value = " ".join(str(value).split())
        if value and value.lower() not in ("none", "nan"):
            parts.append(field + ": " + value)
    return "\n".join(parts)[:MAX_CHARS]


def load_split(task, split):
    """All trials of one split of the benchmark: [{"id", "text", "y"}], y binary or, for numeric targets, a number."""
    task = TASKS[task] if isinstance(task, str) else task
    out = []
    if task.benchmark == "TrialPanorama":
        path = _files(TRIALPANORAMA, f"{task.source}/{split}.jsonl")[0]
        for i, row in enumerate(_jsonl(path)):
            if task.name == "completion":
                y = str((row.get("labels") or {}).get("outcome") or "")
                if y in ("terminate", "complete"):
                    out.append({"id": f"{split}-{i}", "text": protocol_text(row)[:MAX_CHARS], "y": int(y == "terminate")})
            else:
                y = row.get("answer")
                if isinstance(y, (int, float)) and y > 0:
                    out.append({"id": f"{split}-{i}", "text": str(row.get("question") or "")[:MAX_CHARS], "y": float(y)})
        return out
    import pandas as pd
    frame = pd.read_parquet(_files(TRIALBENCH, f"{task.source}/{split}-*.parquet")[0])
    for i, row in frame.iterrows():
        if task.label == "binary":
            label = str(row.get("Label_Class", "")).strip().lower()
            if label not in ("yes", "no"):
                continue
            y = int(label == "yes")
        else:
            value = row.get("Label_Regre")
            if value is None or not np.isfinite(float(value)):
                continue
            y = float(value)
        text = _trialbench_text(row)
        if len(text) >= 80:
            out.append({"id": f"{split}-{i}", "text": text, "y": y})
    return out


def build(task, n_train=N_TRAIN, n_test=N_TEST):
    """The training and test trials of a task. Numeric targets are binarised at the median of the training trials."""
    task = TASKS[task] if isinstance(task, str) else task
    train, test = load_split(task, "train"), load_split(task, "test")
    rs = np.random.RandomState(SPLIT_SEED)
    if len(train) > n_train:
        train = [train[i] for i in rs.permutation(len(train))[:n_train]]
    if len(test) > n_test:
        test = [test[i] for i in rs.permutation(len(test))[:n_test]]
    out = {"task": task.name, "label": task.label, "train": train, "test": test}
    if task.label == "above_median":
        out["median"] = median = float(np.median([r["y"] for r in train]))
        for r in train + test:
            r["value"], r["y"] = r["y"], int(r["y"] > median)
    return out


def evaluation_subset(test, n=N_EVAL):
    """Positions of the test protocols on which the audit is evaluated."""
    return np.sort(np.random.RandomState(EVAL_SEED).permutation(len(test))[:n])


def save(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def load(task, root="data"):
    """The file written by scripts/prepare_data.py."""
    path = os.path.join(root, f"{task}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python scripts/prepare_data.py --task {task}` first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
