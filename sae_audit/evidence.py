"""Evidence E(x): what the concept probe has measured about the concepts that a protocol contains.

The evidence is text, computed from the SAE concepts present in the protocol and from statistics of the training
trials. No LLM is called. It has three parts (Section 2 and Appendix A of the paper):

  concept lines    the concepts of the probe that are present in the protocol, each with its name, the outcome it
                   is associated with, and the size of the association (its effect);
  aggregate        how training trials with a similar set of concepts ended;
  similar trials   the most similar training trials with their outcomes (settings without the auditor).

`EvidenceModel` holds everything that is measured once per task on the training trials.
"""
from collections import Counter

import numpy as np

MIN_MCC = 0.7                   # a feature is a concept if its name was verified at this MCC
MIN_SUPPORT, MAX_SUPPORT = 0.01, 0.5      # and it is present in 1-50% of the training protocols
PSEUDO_COUNT = 2.0
SPLIT_SEED = 17                 # the training trials are split in two halves: one ranks concepts, one measures effects
MIN_TRIALS = 20                 # an effect is reported only if at least this many trials contain the concept
N_MARKERS = 12                  # concepts most strongly associated with each outcome
HEADINGS = ("Of the task", "Recorded", "The record", "The most similar", "The neighborhood")
SUMMARY_SYSTEM = ("You write concise clinical task hypotheses grounded strictly in the marker lists shown. No numbers, "
                  "no percentages, no invented markers.")


def presence(feature_lists, concept_ids):
    """Binary matrix: row i, column j is 1 if concept j is among the features present in protocol i."""
    col = {int(c): j for j, c in enumerate(concept_ids)}
    X = np.zeros((len(feature_lists), len(concept_ids)), np.float32)
    for i, feats in enumerate(feature_lists):
        for f in feats:
            j = col.get(int(f))
            if j is not None:
                X[i, j] = 1.0
    return X


def strength(effect):
    """The fixed verbal scale for the size of an effect."""
    d = abs(effect)
    return "overwhelmingly" if d >= 0.25 else "clearly" if d >= 0.12 else "modestly" if d >= 0.06 else "only slightly"


class EvidenceModel:
    """Statistics of one task, measured on its training trials.

    task             a `sae_audit.tasks.Task`
    train_features   for every training trial, the ids of the SAE features present in its protocol
    train_y          binary outcomes of the training trials
    probe            the concept probe of the task: {"concepts": feature ids, "weights": ..., "names": ...}
    vocabulary       {feature id: {"name", "mcc", ...}} for the named features
    descriptions     optional {feature id: {"name", "description", "long"}}: the name and the one-sentence
                     description with which a concept is shown in the evidence
    train_texts      texts of the training protocols, needed only for the similar trials
    """

    def __init__(self, task, train_features, train_y, probe, vocabulary, descriptions=None, train_texts=None):
        self.task = task
        self.descriptions = {int(k): v for k, v in (descriptions or {}).items()}
        self.train_texts = train_texts
        y = self.y = np.asarray(train_y, float)
        n = len(y)
        self.n_train = n
        self.base = p1 = float(y.mean())

        # all verified concepts that are present in 1-50% of the training protocols: used for similarity between protocols
        support = Counter()
        for feats in train_features:
            support.update({int(f) for f in feats})
        self.pool = np.array(sorted(f for f, c in support.items()
                                    if MIN_SUPPORT * n <= c <= MAX_SUPPORT * n
                                    and float(vocabulary.get(int(f), {}).get("mcc", 0)) >= MIN_MCC))
        Xtr = presence(train_features, self.pool)
        n1, n0 = Xtr.T @ y, Xtr.T @ (1 - y)
        self.log_odds = np.log((n1 + PSEUDO_COUNT) / (n0 + PSEUDO_COUNT)) - np.log(p1 / (1 - p1))
        Ztr = Xtr * np.abs(self.log_odds)[None, :]
        self._train_unit = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)

        # the concepts of the probe
        self.concepts = np.array(probe["concepts"])
        self.weights = np.array(probe["weights"], float)
        self.names = list(probe["names"])
        A = presence(train_features, self.concepts)
        perm = np.random.RandomState(SPLIT_SEED).permutation(n)
        rank_half, effect_half = perm[:n // 2], perm[n // 2:]

        # effect of each concept: outcome rate among the trials that contain it, on the half not used for ranking
        Ae, ye = A[effect_half], y[effect_half]
        self.n_with = Ae.sum(0)
        self.rate = np.array([float(ye[Ae[:, j] > 0].mean()) if self.n_with[j] >= MIN_TRIALS else p1
                              for j in range(len(self.concepts))])

        # the concepts most strongly associated with each outcome (twelve per outcome), ranked on the other half
        Ar, yr = A[rank_half], y[rank_half]
        c1, c0 = Ar[yr == 1].sum(0), Ar[yr == 0].sum(0)
        self.rank = (np.log((c1 + 2.0) / (float(yr.sum()) + 4.0)) - np.log((c0 + 2.0) / (float((1 - yr).sum()) + 4.0)))
        usable = (Ar.sum(0) >= 80) & (self.n_with >= MIN_TRIALS)

        def strongest(order):
            kept = []
            for j in order:
                if not usable[j]:
                    continue
                if all(_phi(Ar[:, j], Ar[:, k]) <= 0.5 for k in kept):     # skip a concept that duplicates a kept one
                    kept.append(j)
                if len(kept) >= N_MARKERS:
                    break
            return kept
        self.pos_markers, self.neg_markers = strongest(_order(-self.rank)), strongest(_order(self.rank))

        # outcome rate of the training trials by their balance of the two kinds of concepts
        net = (Ae[:, self.pos_markers].sum(1) - Ae[:, self.neg_markers].sum(1)).astype(int)
        self.profile = {}
        for b in range(-5, 6):
            m = (net <= -5) if b == -5 else (net >= 5) if b == 5 else (net == b)
            if m.sum() >= MIN_TRIALS:
                self.profile[b] = (int(m.sum()), float(ye[m].mean()))

    # ------------------------------------------------------------------ one protocol
    def read(self, features):
        """Concepts of the probe present in a protocol, and its similarity to every training trial."""
        act = presence([features], self.concepts)[0]
        z = presence([features], self.pool) * np.abs(self.log_odds)[None, :]
        sim = ((z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)) @ self._train_unit.T)[0]
        return act, sim

    def name(self, j, rich=False):
        """The name and the description with which concept j is shown (`rich`: the longer description)."""
        e = self.descriptions.get(int(self.concepts[j]), {})
        nm = str(e.get("name") or self.names[j]).split(" \u2014 ")[0].strip()
        if len(nm) > 100:
            nm = nm[:100].rsplit(" ", 1)[0] + "..."
        return nm, (e.get("long") if rich and e.get("long") else e.get("description", ""))

    def select(self, act, min_lines=4):
        """The concept lines of a protocol: the strongest concepts of each outcome that are present, then further
        present concepts in order of their association with the outcome, up to `min_lines`."""
        m1 = [j for j in self.pos_markers if act[j] > 0]
        m0 = [j for j in self.neg_markers if act[j] > 0]
        picks = [(j, True) for j in m1] + [(j, False) for j in m0]
        if len(picks) < min_lines:
            used = {j for j, _ in picks}
            order = _order(-np.abs(self.rank) * act)
            more = [j for j in order if act[j] > 0 and j not in used and self.n_with[j] >= MIN_TRIALS]
            picks += [(j, bool(self.rank[j] > 0)) for j in more[:min_lines - len(picks)]]
        return picks, m1, m0

    def similar_trials(self, sim, k=4, inverted=False, balanced=False):
        """The k most similar training trials with their outcomes (`balanced`: half from each outcome)."""
        if self.train_texts is None:
            raise ValueError("the most similar trials need the texts of the training trials (`train_texts`)")
        w0, w1 = self.task.outcomes
        order = _order(-sim)
        if balanced:
            half = max(1, k // 2)
            near = sorted([j for j in order if self.y[j] == 1][:half] + [j for j in order if self.y[j] == 0][:half],
                          key=lambda j: -sim[j])
        else:
            near = list(order[:k])
        rows = []
        for a, j in enumerate(near):
            label = 1 - int(self.y[j]) if inverted else int(self.y[j])
            body = " ".join(str(self.train_texts[j]).split())[:300]
            rows.append(f"  CASE {a + 1} [recorded outcome: {w1 if label else w0}]  {body} ...")
        return ("NEAREST RECORDED CASES:\n" + "\n".join(rows) + "\n\n") if rows else ""

    def text(self, features, words=False, rich=False, lines_only=False, neighbours=True, cases=None, min_lines=4,
             inverted=False):
        """The evidence text of one protocol.

        words        effects on the verbal scale ("clearly more often ended in ...") instead of rates
        rich         the longer description of each concept
        lines_only   the concept lines without the aggregate (the form given together with an auditor)
        neighbours   the aggregate also states how the 20 most similar training trials ended
        cases        append the most similar training trials; by default, whenever `train_texts` was given
        inverted     control: every direction, rate and outcome is inverted
        """
        t, (w0, w1), p1 = self.task, self.task.outcomes, self.base
        cases = self.train_texts is not None if cases is None else cases
        act, sim = self.read(features)
        picks, m1, m0 = self.select(act, min_lines)
        L = [f"A frozen evidence model has indexed {self.n_train:,} {t.noun} with known "
             f"outcomes by the clinical concepts each instantiates. Overall, {p1 * 100:.0f}% of "
             f"them ended in {w1} -- rates above that base are evidence toward {w1}, rates "
             f"below it are evidence toward {w0}. THIS protocol shows:"]
        for j, up in picks:
            tag = w1 if ((not up) if inverted else up) else w0
            star = " (a task marker)" if (j in self.pos_markers or j in self.neg_markers) else ""
            r = 1.0 - float(self.rate[j]) if inverted else float(self.rate[j])
            nm, gloss = self.name(j, rich)
            gloss = f" ({gloss.rstrip('.')})" if gloss else ""
            d = r - p1
            if words:
                L.append(f'  [{tag}-leaning{star}] "{nm}"{gloss} -- {t.noun} showing '
                         f"this {strength(d)} more often ended in {w1 if d > 0 else w0}")
            else:
                toward = (f"{abs(d) * 100:.0f} points {'above' if d > 0 else 'below'} the base -- "
                          f"evidence toward {w1 if d > 0 else w0}")
                L.append(f'  [{tag}-leaning{star}] "{nm}"{gloss} -- {r * 100:.0f}% of the '
                         f"{int(self.n_with[j]):,} {t.noun} showing it ended in {w1}, {toward}")
        if lines_only:
            return "\n".join(L) + ("\n\n" + self.similar_trials(sim, inverted=inverted, balanced=not neighbours)
                                   if cases else "")

        c1, c0 = (len(m0), len(m1)) if inverted else (len(m1), len(m0))
        L.append(f"  Of the task's twenty-four strongest markers ({len(self.pos_markers)} {w1}-side, "
                 f"{len(self.neg_markers)} {w0}-side), this protocol shows {c1} {w1}-side and "
                 f"{c0} {w0}-side.")
        net = len(m1) - len(m0)
        b = max(-5, min(5, net))
        if b in self.profile:
            n_like, rate_like = self.profile[b]
            rr = 1.0 - rate_like if inverted else rate_like
            shown = -net if inverted else net
            side = f"{w1}-side by {abs(shown)}" if shown > 0 else f"{w0}-side by {abs(shown)}" if shown < 0 else "balanced"
            d = rr - p1
            if words:
                L.append(f"  Recorded {t.noun} with a profile like this one {strength(d)} "
                         f"more often ended in {w1 if d > 0 else w0}.")
            else:
                L.append(f"  Among the {n_like:,} recorded {t.noun} whose marker "
                         f"profile is likewise {side}, {rr * 100:.0f}% ended in {w1} -- "
                         f"{abs(d) * 100:.0f} points {'above' if d > 0 else 'below'} the base, "
                         f"evidence toward {w1 if d > 0 else w0}.")
            gap = abs(rate_like - p1)
            if gap >= 0.18:
                L.append("  The record's lean on this profile is strong and consistent.")
            elif gap < 0.10:
                L.append("  The record's lean on this profile is mild; weigh it against "
                         "the protocol's specifics rather than over them.")
        if neighbours:
            k20 = int(self.y[_order(-sim)[:20]].sum())
            k20 = 20 - k20 if inverted else k20
            d = k20 / 20.0 - p1
            if words:
                L.append(f"  The most similar recorded cases {strength(d)} more often "
                         f"ended in {w1 if d > 0 else w0}.")
            else:
                L.append(f"  Of the 20 recorded {t.noun} most similar to this one across all "
                         f"indexed concepts, {k20} ended in {w1} -- "
                         f"{abs(d) * 100:.0f} points {'above' if d > 0 else 'below'} the base, "
                         f"evidence toward {w1 if d > 0 else w0}.")
            gap = abs(k20 / 20.0 - p1)
            if gap >= 0.18:
                L.append("  The neighborhood's lean is strong.")
            elif gap < 0.10:
                L.append("  The neighborhood's lean is mild; weigh it against the "
                         "protocol's specifics.")
        return "\n".join(L) + "\n\n" + (self.similar_trials(sim, inverted=inverted, balanced=not neighbours)
                                         if cases else "")

    def summary_prompt(self):
        """The prompt of the task summary: three sentences, written once per task by an LLM from the eight concepts
        most strongly associated with each outcome."""
        w0, w1 = self.task.outcomes

        def line(j):
            e = self.descriptions.get(int(self.concepts[j]), {})
            text = e.get("description", "")
            return "  - " + (e.get("name") or str(self.names[j])[:70]) + (" (" + text.rstrip(".") + ")" if text else "")
        pos, neg = ("\n".join(line(j) for j in side[:8]) for side in (self.pos_markers, self.neg_markers))
        return SUMMARY_SYSTEM, (
            "Registered clinical-trial protocols are indexed by concepts. Trials that ended in "
            "%s typically show these markers:\n%s\n\nTrials that ended in %s typically show "
            "these:\n%s\n\nIn THREE sentences, state the task hypothesis: what kind of trial "
            "tends to end in %s, what kind in %s, and what a reader should look for in a new "
            "protocol. Ground every claim in the markers above." % (w1, pos, w0, neg, w1, w0))

    def aggregate(self, features, inverted=False):
        """The aggregate alone, on the verbal scale: the form given to the auditor."""
        full = self.text(features, words=True, rich=True, cases=False, inverted=inverted)
        return "\n".join(ln.strip() for ln in full.split("\n") if ln.strip().startswith(HEADINGS))


def _order(values):
    """Indices that sort `values` in ascending order; equal values keep their original order."""
    return np.argsort(values, kind="stable")


def _phi(a, b):
    """Phi coefficient of two binary columns."""
    n11, n1, n_1, n = float((a * b).sum()), float(a.sum()), float(b.sum()), float(len(a))
    den = (n1 * n_1 * (n - n1) * (n - n_1)) ** 0.5
    return ((n11 * n - n1 * n_1) / den) if den else 0.0
