"""The released artifacts: concept vocabulary, concept probes, task summaries and encoded protocols.

  artifacts/vocabulary.json            the 11,082 named SAE features with their MCC; 1,772 are verified concepts
  artifacts/descriptions.json          name and one-sentence description of the concepts shown in the evidence
  artifacts/probes/{task}.json         the concept probe of each task
  artifacts/task_summaries/{task}.txt  the task summary of each task
  artifacts/evaluation_ids.json        the 250 test protocols per task on which the audit is evaluated
  artifacts/features/{task}.npz        the verified concepts present in every training and test protocol, with outcomes

Set SAE_AUDIT_ARTIFACTS to read them from another directory.
"""
import json
import os

import numpy as np

ROOT = os.environ.get("SAE_AUDIT_ARTIFACTS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                             "artifacts")


def _json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def vocabulary():
    """{feature id: {"name", "mcc", "clinical", ...}} for all named features."""
    return {int(e["feature"]): e for e in _json("vocabulary.json")["concepts"]}


def descriptions():
    return {int(k): v for k, v in _json("descriptions.json").items()}


def probe(task):
    return _json("probes", f"{task}.json")


def summary(task):
    with open(os.path.join(ROOT, "task_summaries", f"{task}.txt"), encoding="utf-8") as fh:
        return fh.read().strip()


def evaluation_ids(task):
    return _json("evaluation_ids.json")[task]


def save_features(path, concepts, splits):
    """splits: {"train": (ids, feature lists, outcomes), "test": (...)}. Only features in `concepts` are stored."""
    column = {int(c): j for j, c in enumerate(concepts)}
    arrays = {"concepts": np.asarray(concepts, np.int32)}
    for name, (ids, lists, y) in splits.items():
        rows = [np.sort(np.array([column[int(f)] for f in feats if int(f) in column], np.uint16)) for feats in lists]
        arrays[f"{name}_ids"] = np.asarray(ids)
        arrays[f"{name}_y"] = np.asarray(y, np.uint8)
        arrays[f"{name}_indptr"] = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        arrays[f"{name}_indices"] = np.concatenate(rows) if rows else np.zeros(0, np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def features(task, path=None):
    """Encoded protocols of a task: {"train": {"ids", "features", "y"}, "test": {...}}; `features` holds, for every
    protocol, the ids of the verified concepts present in it."""
    with np.load(path or os.path.join(ROOT, "features", f"{task}.npz")) as z:
        data = {k: z[k] for k in z.files}                  # every member is read once
    concepts, out = data["concepts"], {}
    for name in ("train", "test"):
        ptr, idx = data[f"{name}_indptr"], data[f"{name}_indices"]
        out[name] = {"ids": [str(i) for i in data[f"{name}_ids"]], "y": data[f"{name}_y"].astype(int),
                     "features": [concepts[idx[ptr[i]:ptr[i + 1]]] for i in range(len(ptr) - 1)]}
    return out


def evidence_model(task, train_texts=None):
    """The `EvidenceModel` of a task from the released artifacts. `train_texts`: {id: text} of the training trials
    (from `sae_audit.data.load`), needed only to list the most similar trials."""
    from .evidence import EvidenceModel
    from .tasks import TASKS
    train = features(task)["train"]
    texts = [train_texts[i] for i in train["ids"]] if train_texts else None
    return EvidenceModel(TASKS[task], train["features"], train["y"], probe(task), vocabulary(), descriptions(), texts)
