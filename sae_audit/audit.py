"""The audit: evidence E(x) supervises a black-box LLM at inference time through its prompt.

Two settings, which differ in when the LLM sees the evidence (Eqs. 3 and 4 of the paper):

  draft-first      d0 = LLM(x),  d1 = LLM(x, d0, E(x))      the LLM revises its own draft against the evidence
  evidence-first   d1 = LLM(x, E(x))                        the LLM reads the evidence before it answers

In either setting an auditor, the same LLM with its own instructions, can first relate the evidence to the protocol;
its output is added to the inputs of the final call. The prompts are those of the paper (Appendix H).
"""
import re

from .metrics import majority

REVISE = "Incorporate the feedback above and produce a revised analysis."

AUDITOR_SYSTEM = ("You audit a clinical-trial analysis against a frozen concept detector's reading of the same "
                  "protocol. You are impartial: you report where the analysis and the record AGREE as well as where "
                  "they conflict, and the force of your language always matches the recorded strength of the "
                  "evidence.")
AUDITOR_ASK = ("Write an AUDIT of the analysis, 3 to 5 lines, formatted 'A1: ...'. Each line must pair a specific "
               "claim or omission in the analysis (quote its words) with specific concept evidence above (name the "
               "concept and carry its recorded direction and strength wording faithfully). Include what the "
               "analysis gets RIGHT by the record as well as what it conflicts with or ignores. CALIBRATION RULE: "
               "reserve words like 'conflicts' or 'contradicts' for evidence the record marks 'clearly' or stronger; "
               "where the record reads 'only slightly' or 'modestly', present the point as a consideration, not a "
               "contradiction. If the record's overall reading is weak or mixed, state that plainly in your first "
               "line. Do not predict the outcome; do not tell the analyst what to conclude.")
REWRITE = ("Rewrite your analysis to address the audit -- correct what the record contradicts, keep what your own "
           "reading of the protocol genuinely supports, and integrate the named concepts where they sharpen the "
           "reasoning. Output ONLY the improved analysis. Do NOT state a final verdict, answer, or outcome.")

HYPOTHESES_SYSTEM = ("You are a clinical-trial audit writer. A frozen sparse-autoencoder concept detector, trained "
                     "only to index registered protocols, has read this protocol. You write instance hypotheses a "
                     "trialist could check.")
HYPOTHESES_ASK = ("Write INSTANCE HYPOTHESES ('H1: ...'), but ONLY where a detected concept and a quoted protocol "
                  "phrase GENUINELY meet -- the concept's recorded meaning must actually describe that phrase. Never "
                  "force a pairing; never write a hypothesis to balance the other side. If fewer than two genuine "
                  "bindings exist, or the genuine ones conflict, write no further hypotheses and say so. Always close "
                  "with one line 'RECORD READING: ...' restating the aggregate reading above at exactly its recorded "
                  "strength wording (both readings separately if they disagree, never blended). Do not predict the "
                  "outcome; no probabilities.")


# ---------------------------------------------------------------------------------------------- answers
def parse_answer(task, text):
    """The answer stated in an analysis: 1 for the positive outcome, 0 for the negative one, None if there is none."""
    if not text:
        return None
    w0, w1 = task.outcomes
    m = re.findall(rf'"outcome"\s*:\s*"?({w0}|{w1})', text, re.I)
    if m:
        return 1 if m[-1].lower() == w1 else 0
    tail = text[-200:].lower()
    h1, h0 = tail.rfind(w1[:6]), tail.rfind(w0[:6])
    return None if max(h1, h0) < 0 else (1 if h1 > h0 else 0)


def strip_answer(text):
    """An analysis without its stated answer."""
    if not text:
        return ""
    t = re.sub(r'\{[^{}]*"outcome"[^{}]*\}', " ", text)
    t = re.sub(r"```[a-z]*|```", " ", t)
    return " ".join(t.split())


# ---------------------------------------------------------------------------------------------- prompts
def unaided_prompt(task, protocol):
    """The LLM alone. Its answer is also the draft d0 of the draft-first setting."""
    return task.system, f"{protocol[:5000]}\n\n{task.question}"


def _with_summary(heading, summary):
    return (heading + "\n" + summary.strip() + "\n\n") if summary and summary.strip() else ""


def _detected(model, features, inverted):
    return model.text(features, words=True, rich=True, lines_only=True, cases=False, min_lines=10, inverted=inverted)


def auditor_evidence(model, features, summary, inverted=False):
    """E(x) as the auditor receives it: task summary, concept lines, aggregate."""
    return (_with_summary("TASK HYPOTHESIS (from the record):", summary)
            + "CONCEPTS DETECTED in this protocol:\n" + _detected(model, features, inverted)
            + "\n\nTHE DETECTOR'S AGGREGATE READING (measured on the training record):\n"
            + model.aggregate(features, inverted))


def draft_first_prompt(task, model, protocol, features, draft, inverted=False):
    """d1 = LLM(x, d0, E(x)) without the auditor."""
    body = protocol[:5000]
    return task.system, (f"{body}\n\n{task.question}\n\nYOUR PREVIOUS ANALYSIS:\n{draft.strip()}\n\n"
                         f"{model.text(features, inverted=inverted)}{REVISE}")


def evidence_first_prompt(task, model, protocol, features, summary, inverted=False):
    """d1 = LLM(x, E(x)) without the auditor: a single call."""
    body = protocol[:5000]
    evidence = (_with_summary("THE TASK, AS THE RECORD TELLS IT:", summary)
                + model.text(features, words=True, rich=True, inverted=inverted))
    return task.system, f"{body}\n\n{evidence}{task.question}"


def auditor_prompt(task, model, protocol, features, summary, draft=None, inverted=False):
    """a = Auditor(x, d0, E(x)). With a draft the auditor audits it; without one it writes hypotheses about the trial."""
    body, evidence = protocol[:5000][:4000], auditor_evidence(model, features, summary, inverted)
    if draft is None:
        return HYPOTHESES_SYSTEM, f"{body}\n\n{evidence}\n\n{HYPOTHESES_ASK}"
    return AUDITOR_SYSTEM, f"{body}\n\nTHE ANALYSIS UNDER AUDIT:\n{draft.strip()}\n\n{evidence}\n\n{AUDITOR_ASK}"


def audited_prompt(task, model, protocol, features, summary, audit, draft=None, inverted=False):
    """The final call with the auditor's output: d1 = LLM(x, d0, E(x), a)."""
    body = protocol[:5000]
    evidence = (_with_summary("THE TASK, AS THE RECORD TELLS IT:", summary)
                + "CONCEPTS DETECTED (frozen dictionary; ranked by weight of evidence):\n"
                + _detected(model, features, inverted) + "\n\n")
    if draft is None:
        return task.system, (f"{body}\n\n{evidence}INSTANCE HYPOTHESES (from the concept detector's reading of this "
                             f"protocol):\n{audit.strip()}\n\n{task.question}")
    return task.system, (f"{body}\n\nYOUR PREVIOUS ANALYSIS:\n{draft.strip()}\n\n{evidence}"
                         f"AN AUDIT OF YOUR ANALYSIS AGAINST THE RECORD:\n{audit.strip()}\n\n{REWRITE}")


def read_prompt(task, protocol, analysis):
    """A separate call that reads the answer from a rewritten analysis (draft-first with the auditor)."""
    return task.system, f"{protocol[:5000]}\n\nAN ANALYSIS OF THIS TRIAL:\n{strip_answer(analysis)}\n\n{task.question}"


# ---------------------------------------------------------------------------------------------- one protocol
def audit(llm, task, model, protocol, features, summary="", setting="draft_first", auditor=False, runs=3,
          inverted=False, key=None):
    """SAE-Audit on one protocol. Returns the majority answer of the runs and the texts of every run.

    llm        `sae_audit.llm.LLM`
    model      `sae_audit.evidence.EvidenceModel` of the task
    features   ids of the SAE features present in the protocol
    summary    the task summary (used by every setting except draft-first without the auditor)
    key        a string that identifies the protocol in the call cache
    """
    assert setting in ("draft_first", "evidence_first")
    tag = f"{llm.model}|{task.name}|{key}" if key is not None else None
    name = setting + ("+auditor" if auditor else "") + ("|inverted" if inverted else "")

    def call(prompt, step, run, max_tokens=600):
        return llm.chat(*prompt, max_tokens=max_tokens, key=None if tag is None else f"{tag}|{step}|{run}")

    out = []
    for run in range(runs):
        draft = call(unaided_prompt(task, protocol), "unaided", run) if setting == "draft_first" else None
        rec = {"draft": draft, "audit": None, "final": None}
        if setting == "draft_first" and (not draft or len(draft) < 20):
            out.append(rec)
            continue
        if auditor:
            rec["audit"] = call(auditor_prompt(task, model, protocol, features, summary, draft, inverted),
                                f"{name}|auditor", run)
            if rec["audit"]:
                rec["final"] = call(audited_prompt(task, model, protocol, features, summary, rec["audit"], draft,
                                                   inverted), f"{name}|final", run)
            if rec["final"] and draft is not None:          # the rewritten analysis states no answer: read it
                rec["read"] = call(read_prompt(task, protocol, rec["final"]), f"{name}|read", run, max_tokens=300)
        elif setting == "draft_first":
            rec["final"] = call(draft_first_prompt(task, model, protocol, features, draft, inverted), f"{name}|final", run)
        else:
            rec["final"] = call(evidence_first_prompt(task, model, protocol, features, summary, inverted),
                                f"{name}|final", run)
        rec["answer"] = parse_answer(task, rec.get("read") or rec["final"])
        out.append(rec)
    return majority([r.get("answer") for r in out]), out
