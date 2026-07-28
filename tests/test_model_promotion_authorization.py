#!/usr/bin/env python3
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import loto7_model_self_evolver as cli


class ModelPromotionAuthorizationTests(unittest.TestCase):
    def test_automatic_apply_is_removed_without_authorization(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            arguments = cli._authorized_argv(["--iterations", "1", "--apply"])
        self.assertEqual(arguments, ["--iterations", "1"])

    def test_explicit_environment_authorization_retains_apply(self) -> None:
        with patch.dict(
            os.environ, {"LOTO7_ALLOW_LEGACY_DIRECT_APPLY": "1"}, clear=True
        ):
            arguments = cli._authorized_argv(["--iterations", "1", "--apply"])
        self.assertEqual(arguments, ["--iterations", "1", "--apply"])

    def test_standalone_workflow_dispatch_retains_apply(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_WORKFLOW": "LOTO7 Model Self Evolution",
            },
            clear=True,
        ):
            arguments = cli._authorized_argv(["--iterations", "1", "--apply"])
        self.assertEqual(arguments, ["--iterations", "1", "--apply"])

    def test_integrated_workflow_dispatch_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_WORKFLOW": "LOTO7 Evolution Trainer",
            },
            clear=True,
        ):
            arguments = cli._authorized_argv(["--iterations", "1", "--apply"])
        self.assertEqual(arguments, ["--iterations", "1"])

    def test_scheduled_standalone_workflow_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "schedule",
                "GITHUB_WORKFLOW": "LOTO7 Model Self Evolution",
            },
            clear=True,
        ):
            arguments = cli._authorized_argv(["--iterations", "1", "--apply"])
        self.assertEqual(arguments, ["--iterations", "1"])

    def test_arguments_without_apply_are_unchanged(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            arguments = cli._authorized_argv(["--iterations", "2"])
        self.assertEqual(arguments, ["--iterations", "2"])


if __name__ == "__main__":
    unittest.main()
