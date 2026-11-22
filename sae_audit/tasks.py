"""The five outcome-prediction tasks of the paper.

Each task asks one yes/no question about a clinical trial, given the protocol as registered before the trial
started. `outcomes` is (negative, positive): the positive outcome is the label 1 of the probes. For the two numeric
targets (duration, sample size) the label is whether the value exceeds the median of the training trials.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    benchmark: str              # "TrialPanorama" or "TrialBench"
    source: str                 # dataset configuration on the Hugging Face Hub
    label: str                  # "binary" or "above_median"
    system: str                 # system prompt of every call that answers the task
    question: str               # the task question
    outcomes: tuple             # (negative, positive)
    noun: str                   # how the evidence refers to the training trials
    setting: str                # the setting reported in Table 1: "draft_first" or "evidence_first"
    auditor: bool               # whether that setting includes the auditor

    @property
    def negative(self):
        return self.outcomes[0]

    @property
    def positive(self):
        return self.outcomes[1]


def _system(expert, judge, answer):
    return (f"You are a clinical trial {expert} expert. Given a trial's {judge}\n"
            "Give one or two sentences of reasoning, then decide.\n"
            f'End with strict JSON on its own line: {{"outcome":"{answer}"}}')


TASKS = {
    "completion": Task(
        "completion", "TrialPanorama", "trial_completion_assessment", "binary",
        _system("operations", "protocol as registered BEFORE it ran, judge whether it went on to COMPLETE or was "
                "TERMINATED early.", "complete|terminate"),
        "Did this trial complete or terminate early?", ("complete", "terminate"), "registered protocols",
        "draft_first", False),
    "adverse_event": Task(
        "adverse_event", "TrialBench", "adverse_event_prediction_processed", "binary",
        _system("safety", "protocol as registered BEFORE it ran, judge whether it went on to report SERIOUS adverse "
                "events or NONE.", "none|serious"),
        "Did this trial report serious adverse events?", ("none", "serious"), "registered protocols",
        "evidence_first", False),
    "duration": Task(
        "duration", "TrialBench", "trial_duration_prediction_processed", "above_median",
        _system("operations", "protocol as registered BEFORE it ran, judge whether it ran LONGER or SHORTER than a "
                "median registered trial.", "shorter|longer"),
        "Did this trial run longer or shorter than the median?", ("shorter", "longer"), "registered protocols",
        "evidence_first", True),
    "sample_size": Task(
        "sample_size", "TrialPanorama", "sample_size_estimation", "above_median",
        _system("operations", "design as registered BEFORE it ran, judge whether its final enrolment was ABOVE or "
                "BELOW the median enrolment for registered trials.", "below|above"),
        "Was this trial's enrolment above or below the median?", ("below", "above"), "registered trial designs",
        "draft_first", False),
    "dropout": Task(
        "dropout", "TrialBench", "patient_dropout_prediction_processed", "binary",
        _system("operations", "protocol as registered BEFORE it ran, judge whether its patient dropout rate turned "
                "out HIGH or LOW.", "low|high"),
        "Was this trial's patient dropout rate high or low?", ("low", "high"), "registered protocols",
        "evidence_first", False),
}
