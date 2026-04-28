"""Tests for _extract_failure_ids and _gate_passes_with_baseline."""

from __future__ import annotations

import sys
import os

# Add src to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple.server import _extract_failure_ids, _gate_passes_with_baseline


# ---------------------------------------------------------------------------
# _extract_failure_ids — pytest
# ---------------------------------------------------------------------------

class TestExtractFailureIdsPytest:
    def test_failed_line(self):
        output = "FAILED tests/foo.py::test_bar - AssertionError"
        ids = _extract_failure_ids(output)
        assert "FAILED tests/foo.py::test_bar" in ids

    def test_failed_line_no_reason(self):
        output = "FAILED tests/foo.py::test_bar"
        ids = _extract_failure_ids(output)
        assert "FAILED tests/foo.py::test_bar" in ids

    def test_error_collecting(self):
        output = "ERROR collecting tests/test_diff.py"
        ids = _extract_failure_ids(output)
        assert "ERROR collecting tests/test_diff.py" in ids

    def test_error_generic(self):
        output = "ERROR tests/test_something.py"
        ids = _extract_failure_ids(output)
        assert "ERROR tests/test_something.py" in ids

    def test_error_bang_not_matched(self):
        """ERROR! lines should not be captured."""
        output = "ERROR! some crash info"
        ids = _extract_failure_ids(output)
        assert len(ids) == 0

    def test_multiple_failures(self):
        output = (
            "FAILED tests/a.py::test_one - reason\n"
            "FAILED tests/b.py::test_two - reason\n"
            "ERROR tests/c.py\n"
        )
        ids = _extract_failure_ids(output)
        assert len(ids) == 3

    def test_rerun_line(self):
        """RERUN lines are normalised to FAILED for stable baseline comparison."""
        output = "RERUN test/units/module_utils/urls/test_Request.py::test_open_url - AssertionError"
        ids = _extract_failure_ids(output)
        assert "FAILED test/units/module_utils/urls/test_Request.py::test_open_url" in ids

    def test_rerun_line_no_reason(self):
        output = "RERUN tests/foo.py::test_bar"
        ids = _extract_failure_ids(output)
        assert "FAILED tests/foo.py::test_bar" in ids

    def test_rerun_without_node_id_ignored(self):
        """RERUN lines without :: (not test IDs) are ignored."""
        output = "RERUN some random message"
        ids = _extract_failure_ids(output)
        assert len(ids) == 0


# ---------------------------------------------------------------------------
# _extract_failure_ids — go test
# ---------------------------------------------------------------------------

class TestExtractFailureIdsGoTest:
    def test_go_fail(self):
        output = "--- FAIL: TestNewPassword (0.00s)"
        ids = _extract_failure_ids(output)
        assert "--- FAIL: TestNewPassword" in ids

    def test_gocheck(self):
        output = "FAIL: password_test.go:104: PasswordSuite.TestTiming"
        ids = _extract_failure_ids(output)
        assert "FAIL: password_test.go:104: PasswordSuite.TestTiming" in ids

    def test_package_fail_tab(self):
        output = "FAIL\tgithub.com/gravitational/teleport/lib/auth\t1.23s"
        ids = _extract_failure_ids(output)
        assert "FAIL github.com/gravitational/teleport/lib/auth" in ids

    def test_package_fail_spaces(self):
        output = "FAIL    github.com/org/repo/pkg    0.5s"
        ids = _extract_failure_ids(output)
        assert "FAIL github.com/org/repo/pkg" in ids


# ---------------------------------------------------------------------------
# _extract_failure_ids — mocha/jest/npm
# ---------------------------------------------------------------------------

class TestExtractFailureIdsMochaJest:
    def test_mocha_numbered(self):
        output = "1) suite name > test name:"
        ids = _extract_failure_ids(output)
        assert "1) suite name > test name" in ids

    def test_mocha_numbered_no_colon(self):
        output = "3) another test"
        ids = _extract_failure_ids(output)
        assert "3) another test" in ids

    def test_sh_not_found(self):
        output = "sh: 1: jest: not found"
        ids = _extract_failure_ids(output)
        assert "sh: 1: jest: not found" in ids


# ---------------------------------------------------------------------------
# _extract_failure_ids — crashes and timeouts
# ---------------------------------------------------------------------------

class TestExtractFailureIdsCrashTimeout:
    def test_fatal_python_error(self):
        output = "Fatal Python error: Aborted\nCurrent thread ..."
        ids = _extract_failure_ids(output)
        assert "Fatal Python error" in ids

    def test_command_timed_out(self):
        output = "[command timed out after 300s]"
        ids = _extract_failure_ids(output)
        assert "[command timed out after 300s]" in ids

    def test_timeout_monitored_command(self):
        output = "timeout: the monitored command dumped core"
        ids = _extract_failure_ids(output)
        assert "timeout: the monitored command dumped core" in ids


# ---------------------------------------------------------------------------
# _extract_failure_ids — mixed real output
# ---------------------------------------------------------------------------

class TestExtractFailureIdsRealOutput:
    def test_ansible_collection_error(self):
        """Real ansible-a26c pattern: ImportError at collection time."""
        output = (
            "============================= ERRORS ==============================\n"
            "ERROR collecting test/units/modules/network/test_diff.py\n"
            " ImportError: cannot import name 'AnsibleModule'\n"
            "=========================== short test summary info ================\n"
            "ERROR test/units/modules/network/test_diff.py\n"
            "!!!!!!!!!!!!!!!!!!!!!!! Errors during collection !!!!!!!!!!!!!!!!!\n"
        )
        ids = _extract_failure_ids(output)
        assert any("test_diff.py" in i for i in ids)
        assert len(ids) >= 1

    def test_empty_output(self):
        assert _extract_failure_ids("") == set()

    def test_normal_pass_output(self):
        output = "42 passed in 0.20s\n"
        assert _extract_failure_ids(output) == set()


# ---------------------------------------------------------------------------
# _gate_passes_with_baseline — basic logic
# ---------------------------------------------------------------------------

class TestGatePassesWithBaseline:
    def test_gate_passes_exit_zero(self):
        """Exit code 0 always passes."""
        passes, new, base = _gate_passes_with_baseline(0, "", None, "")
        assert passes is True
        assert new == set()

    def test_gate_fails_no_baseline(self):
        """No baseline → fall back to exit code."""
        passes, new, base = _gate_passes_with_baseline(
            1, "FAILED tests/foo.py::test_bar", None, "",
        )
        assert passes is False
        assert len(new) == 1

    def test_gate_fails_clean_baseline(self):
        """Baseline exit 0 → no pre-existing failures → gate fails."""
        passes, new, base = _gate_passes_with_baseline(
            1, "FAILED tests/foo.py::test_bar", 0, "42 passed",
        )
        assert passes is False

    def test_same_failures_as_baseline(self):
        """All gate failures also in baseline → passes."""
        baseline_out = (
            "FAILED tests/foo.py::test_bar - ImportError\n"
            "ERROR tests/baz.py\n"
        )
        gate_out = (
            "FAILED tests/foo.py::test_bar - ImportError\n"
            "ERROR tests/baz.py\n"
        )
        passes, new, base = _gate_passes_with_baseline(1, gate_out, 1, baseline_out)
        assert passes is True
        assert new == set()
        assert len(base) == 2

    def test_subset_of_baseline_passes(self):
        """Gate has fewer failures than baseline → passes (no regressions)."""
        baseline_out = (
            "FAILED tests/a.py::test_one - reason\n"
            "FAILED tests/b.py::test_two - reason\n"
        )
        gate_out = "FAILED tests/a.py::test_one - reason\n"
        passes, new, base = _gate_passes_with_baseline(1, gate_out, 1, baseline_out)
        assert passes is True
        assert new == set()

    def test_new_failure_detected(self):
        """Gate has a failure not in baseline → fails."""
        baseline_out = "FAILED tests/a.py::test_one - reason\n"
        gate_out = (
            "FAILED tests/a.py::test_one - reason\n"
            "FAILED tests/b.py::test_new - reason\n"
        )
        passes, new, base = _gate_passes_with_baseline(1, gate_out, 1, baseline_out)
        assert passes is False
        assert "FAILED tests/b.py::test_new" in new

    def test_both_fail_no_extractable_ids(self):
        """Both fail but no parseable failure IDs → pass permissively (opaque baseline)."""
        passes, new, base = _gate_passes_with_baseline(
            1, "some random output", 1, "some other random output",
        )
        assert passes is True
        assert new == set()
        assert base == set()

    def test_go_test_baseline_filtering(self):
        """Go test failures filtered against baseline."""
        baseline_out = "--- FAIL: TestOldBug (0.01s)\nFAIL\tgithub.com/org/repo/pkg\t0.5s"
        gate_out = "--- FAIL: TestOldBug (0.02s)\nFAIL\tgithub.com/org/repo/pkg\t0.6s"
        passes, new, base = _gate_passes_with_baseline(1, gate_out, 1, baseline_out)
        assert passes is True
        assert new == set()

    def test_ansible_a26c_scenario(self):
        """The exact ansible-a26c scenario: collection error in baseline and gate."""
        baseline_out = (
            "ERROR collecting test/units/modules/network/test_diff.py\n"
            "ERROR test/units/modules/network/test_diff.py\n"
        )
        gate_out = (
            "ERROR collecting test/units/modules/network/test_diff.py\n"
            "ERROR test/units/modules/network/test_diff.py\n"
        )
        passes, new, base = _gate_passes_with_baseline(2, gate_out, 2, baseline_out)
        assert passes is True
        assert new == set()
        assert len(base) >= 1

    def test_rerun_failures_detected_as_new(self):
        """RERUN failures in gate output not in baseline are caught (log125 scenario).

        With -rfE -p no:rerunfailures, failures appear as FAILED.
        This test covers the defensive RERUN parsing path.
        """
        baseline_out = "ERROR test/ansible_test/unit/test_diff.py\n"
        gate_out = (
            "ERROR test/ansible_test/unit/test_diff.py\n"
            "RERUN test/units/module_utils/urls/test_Request.py::test_open_url - AssertionError\n"
        )
        passes, new, base = _gate_passes_with_baseline(1, gate_out, 1, baseline_out)
        assert passes is False
        assert "FAILED test/units/module_utils/urls/test_Request.py::test_open_url" in new

    def test_rerun_failures_in_both_are_filtered(self):
        """RERUN failures present in both baseline and gate are filtered."""
        baseline_out = (
            "ERROR test/diff.py\n"
            "RERUN test/units/test_Request.py::test_open_url - reason\n"
        )
        gate_out = (
            "ERROR test/diff.py\n"
            "RERUN test/units/test_Request.py::test_open_url - reason\n"
        )
        passes, new, base = _gate_passes_with_baseline(1, gate_out, 1, baseline_out)
        assert passes is True
        assert new == set()
