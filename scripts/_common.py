"""Shared by the scripts: make the package importable from a checkout and list the tasks."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sae_audit.tasks import TASKS                                        # noqa: E402


def tasks(value):
    return list(TASKS) if value == "all" else [value]


def add_task(parser, default=None):
    parser.add_argument("--task", choices=list(TASKS) + ["all"], default=default, required=default is None)
