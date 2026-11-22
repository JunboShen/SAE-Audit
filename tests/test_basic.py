"""Prompts and call sequence of the settings, with a stand-in for the LLM; no API call and no GPU."""
import pytest

from sae_audit import artifacts, audit
from sae_audit.metrics import balanced_accuracy, majority
from sae_audit.probe import predict
from sae_audit.selection import select
from sae_audit.tasks import TASKS
from sae_audit.vocabulary import mcc

TASK = TASKS["completion"]
PROTOCOL = "Title: A study\n\nDesign: allocation=RANDOMIZED\n\nEligibility (gender=ALL):\nAdults with asthma"


class Scripted:
    """Answers every call with a fixed analysis and keeps the prompts."""
    model = "scripted"

    def __init__(self):
        self.prompts = []

    def chat(self, system, user, max_tokens=600, key=None):
        self.prompts.append((system, user))
        return 'The design is small and narrow.\n{"outcome":"terminate"}'


@pytest.fixture(scope="module")
def setup():
    test = artifacts.features("completion")["test"]
    return artifacts.evidence_model("completion"), test["features"][0], artifacts.summary("completion")


def test_parse_answer():
    assert audit.parse_answer(TASK, 'x {"outcome":"terminate"}') == 1
    assert audit.parse_answer(TASK, "I expect the trial to complete.") == 0
    assert audit.parse_answer(TASK, "no decision") is None
    assert audit.strip_answer('Short trial.\n```json\n{"outcome":"complete"}\n```') == "Short trial."


@pytest.mark.parametrize("setting,auditor,calls", [("draft_first", False, 2), ("draft_first", True, 4),
                                                   ("evidence_first", False, 1), ("evidence_first", True, 2)])
def test_calls_per_run(setup, setting, auditor, calls):
    model, features, summary = setup
    llm = Scripted()
    answer, runs = audit.audit(llm, TASK, model, PROTOCOL, features, summary, setting, auditor, runs=1)
    assert answer == 1 and len(llm.prompts) == calls
    assert all(user.startswith(PROTOCOL) for _, user in llm.prompts)


def test_prompts(setup):
    model, features, summary = setup
    system, user = audit.draft_first_prompt(TASK, model, PROTOCOL, features, "My draft.")
    assert system == TASK.system and user.endswith(audit.REVISE)
    assert user.startswith(f"{PROTOCOL}\n\n{TASK.question}\n\nYOUR PREVIOUS ANALYSIS:\nMy draft.\n\n")
    system, user = audit.evidence_first_prompt(TASK, model, PROTOCOL, features, summary)
    assert system == TASK.system and user.endswith(TASK.question)


def test_evidence(setup):
    model, features, _ = setup
    lines = [ln for ln in model.text(features).split("\n") if ln.startswith("  [")]
    assert len(lines) >= 4 and all("-leaning" in ln for ln in lines)


@pytest.mark.parametrize("task", list(TASKS))
def test_probe(task):
    test = artifacts.features(task)["test"]
    probability, label = predict(artifacts.probe(task), test["features"])
    assert balanced_accuracy(label, test["y"]) > 0.6 and 0 <= probability.min() <= probability.max() <= 1


def test_small_functions():
    assert majority([1, None, 1]) == 1 and majority([1, 0]) == 0 and majority([None], 1) == 1
    assert balanced_accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx((1 + 2 / 3) / 2)
    assert select([1, 1, 1], [0.1, 0.2, 0.3]) == 1
    assert select([1, 0, 0, 1], [0.9, -0.2, 0.1, 0.5]) == 1
    assert mcc([1, 1, 1, 1], [1, 1, 0, 0]) == 0.0 and mcc([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
