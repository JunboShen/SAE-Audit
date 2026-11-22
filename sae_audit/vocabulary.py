"""The concept vocabulary: SAE features with a name that a second LLM can apply.

Passages of past protocols (eligibility criteria) are encoded once and divided by trial into two halves. On the first
half an LLM sees passages on which a feature is active and passages on which it is not, and proposes two names. On
the second half a second LLM receives only the names and twelve held-out passages, six active and six inactive, and
lists the passages that have the named property. The score of a name is the Matthews correlation coefficient (MCC)
between that list and the activations of the feature. The better name is kept, and a feature is a concept if its
name reaches an MCC of 0.7. No outcome label is used.
"""
import json
import math
import re

import numpy as np

SEED = 0
MIN_PASSAGES = 30            # a candidate feature is active on at least this many passages of each half
MAX_RATE = 0.25              # and on at most this fraction of the passages of each half
MAX_FEATURES = 12000
MIN_MCC = 0.7

NAME_SYSTEM = (
    "You are naming one feature of a sparse autoencoder over clinical-trial eligibility criteria.\n"
    "You are shown the text spans where it fires most strongly, and spans where it does not fire.\n"
    "Give TWO different candidate descriptions of the property the firing spans have and the others lack.\n"
    "Each must be ONE short phrase, as SPECIFIC as you can make it while still true of every firing span. "
    "A phrase like 'requires a specific condition' is useless because it is true of nearly every trial: "
    "name the actual requirement, restriction, population, measurement, or manner of writing.\n"
    'Output strict JSON on one line: {"a": "<phrase>", "b": "<phrase>"}')

VERIFY_SYSTEM = (
    "You are given two PROPERTIES and numbered clinical-trial text spans.\n"
    "For each property, list the spans that have it, using only the property as stated.\n"
    'Output strict JSON on one line: {"a": [numbers], "b": [numbers]}')


def passages(rows, n=20000):
    """Eligibility-criterion passages of TrialPanorama protocols: [(trial id, passage)], 45 to 320 characters each."""
    out, seen = [], set()
    for row in rows:
        trial = str((row.get("metadata") or {}).get("study_nctid") or len(out))
        criteria = ""
        for c in row.get("context") or []:
            if c.get("type") == "eligibility_criteria":
                criteria = str(c.get("criteria") or "")
        for piece in re.split(r"[~\n]\s*|(?<=\.)\s{2,}", criteria):
            s = re.sub(r"^\s*[\*\-•\d\.\)]+\s*", "", piece).strip()
            if not (45 <= len(s) <= 320) or s.lower() in seen:
                continue
            if re.match(r"(?i)(key )?(inclusion|exclusion) criteria", s):
                continue
            seen.add(s.lower())
            out.append((trial, s))
        if len(out) >= n:
            break
    return out[:n]


def excerpt(text, token, width=9):
    """Up to 19 words of a passage around the token on which a feature is most active."""
    words = text.split()
    if not words:
        return text[:160]
    i = int(np.clip(token * len(words) / max(1, token + 12), 0, len(words) - 1)) if token else 0
    lo, hi = max(0, i - width), min(len(words), i + width + 1)
    return ("... " if lo else "") + " ".join(words[lo:hi]) + (" ..." if hi < len(words) else "")


def mcc(pred, gold):
    pred, gold = np.asarray(pred), np.asarray(gold)
    tp, tn = int(((pred == 1) & (gold == 1)).sum()), int(((pred == 0) & (gold == 0)).sum())
    fp, fn = int(((pred == 1) & (gold == 0)).sum()), int(((pred == 0) & (gold == 1)).sum())
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def _json(text):
    try:
        return json.loads(re.search(r"\{.*\}", text or "", re.S).group())
    except (AttributeError, ValueError):
        return None


def build(trials, texts, encoded, name_llm, verify_llm, max_features=MAX_FEATURES, log=print):
    """Name and verify the features of a set of encoded passages.

    trials, texts   trial id and text of every passage
    encoded         output of `Encoder.encode(texts)`
    name_llm        `sae_audit.llm.LLM` that proposes names
    verify_llm      `sae_audit.llm.LLM` that applies them to held-out passages
    Returns one entry per named feature: {"feature", "name", "mcc", "rate", "examples"}.
    """
    n = len(texts)
    trial_ids = np.unique(trials)
    first = set(trial_ids[np.random.RandomState(SEED).permutation(len(trial_ids))[:len(trial_ids) // 2]])
    in_a = np.array([t in first for t in trials])
    active = ({}, {})                                            # feature -> passages on which it is active, per half
    for i in range(n):
        for f in encoded[i][0]:
            active[0 if in_a[i] else 1].setdefault(int(f), []).append(i)
    size = (int(in_a.sum()), int((~in_a).sum()))
    index = (np.where(in_a)[0], np.where(~in_a)[0])
    members = tuple({f: set(v) for f, v in half.items()} for half in active)
    token = [dict(zip(encoded[i][0].tolist(), encoded[i][2].tolist())) for i in range(n)]

    candidates = [f for f in active[0]
                  if len(active[0][f]) >= MIN_PASSAGES and len(active[1].get(f, [])) >= MIN_PASSAGES
                  and len(active[0][f]) / size[0] <= MAX_RATE and len(active[1][f]) / size[1] <= MAX_RATE]
    candidates = sorted(candidates, key=lambda f: -min(len(active[0][f]), len(active[1][f])))[:max_features]
    log(f"[vocabulary] {n:,} passages of {len(trial_ids):,} trials, {len(candidates):,} candidate features")

    def shown(f, half, k=6):
        """The k longest passages on which f is active, as excerpts, and k random passages on which it is not."""
        longest = sorted(active[half][f], key=lambda i: -len(texts[i]))[:k]
        on = [excerpt(str(texts[i]), int(token[i].get(int(f), 0))) for i in longest]
        off, rng = [], np.random.RandomState(SEED + int(f) % 9973)
        for i in rng.permutation(len(index[half])):
            j = int(index[half][i])
            if j not in members[half][f]:
                off.append(str(texts[j])[:170])
                if len(off) >= k:
                    break
        return on, off

    entries = []
    for f in candidates:
        on, off = shown(f, 0)
        raw = name_llm.chat(NAME_SYSTEM, "\n".join(f"FIRES {i + 1}: {s}" for i, s in enumerate(on)) + "\n\n"
                            + "\n".join(f"DOES NOT FIRE {i + 1}: {s}" for i, s in enumerate(off)),
                            max_tokens=200, key=f"name|{f}")
        names = _json(raw)
        if not names:
            continue
        names = [str(names.get("a", "")).strip(), str(names.get("b", "")).strip()]
        if not any(names):
            continue
        on, off = shown(f, 1)
        held = [(s, 1) for s in on] + [(s, 0) for s in off]
        np.random.RandomState(SEED + f % 997).shuffle(held)
        raw = verify_llm.chat(VERIFY_SYSTEM, f"PROPERTY a: {names[0]}\nPROPERTY b: {names[1]}\n\n"
                              + "\n".join(f"{i + 1}. {s}" for i, (s, _) in enumerate(held)),
                              max_tokens=250, key=f"verify|{f}")
        picked = _json(raw)
        if not picked:
            continue
        gold = np.array([g for _, g in held])
        scored = []
        for key, name in zip("ab", names):
            try:
                chosen = {int(x) for x in (picked.get(key) or [])}
            except (TypeError, ValueError):
                chosen = set()
            scored.append((mcc(np.array([int(i + 1 in chosen) for i in range(len(held))]), gold), name))
        score, name = max(scored, key=lambda t: t[0])
        if name:
            entries.append({"feature": int(f), "name": name, "mcc": round(score, 3),
                            "rate": round(len(active[0][f]) / size[0], 4), "examples": shown(f, 0, 2)[0]})
    entries.sort(key=lambda e: -e["mcc"])
    log(f"[vocabulary] {len(entries):,} named features, {sum(e['mcc'] >= MIN_MCC for e in entries):,} with MCC >= {MIN_MCC}")
    return entries
