"""The selection step, used with GPT-5.6-terra in the paper.

When the drafts and the revisions of the runs do not all give the same answer, keep the answer whose analyses are
more consistent with the concepts they mention. An analysis is encoded with the same SAE; among the concepts that
are present both in the analysis and in the protocol, its consistency is the signed fraction of the association with
the outcome that points towards its own answer. No LLM is called.
"""
import numpy as np

from .evidence import presence


def consistency(model, analysis_features, protocol_features, answer):
    """Between -1 and 1: positive if the concepts mentioned by the analysis are associated with its answer."""
    shared = presence([analysis_features], model.pool)[0] * presence([protocol_features], model.pool)[0]
    total = float((shared * np.abs(model.log_odds)).sum())
    return (2 * answer - 1) * float(shared @ model.log_odds) / total if total > 1e-9 else 0.0


def select(answers, scores):
    """Answers (0/1) and consistency scores of all drafts and revisions of a protocol -> the selected answer."""
    answers, scores = np.asarray(answers), np.asarray(scores, float)
    if not len(answers):
        return None
    if 0 < answers.mean() < 1:
        return int(scores[answers == 1].mean() > scores[answers == 0].mean())
    return int(answers.mean() > 0.5)
