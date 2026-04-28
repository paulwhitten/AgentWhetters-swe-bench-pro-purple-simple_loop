"""Unit tests for DockerRunner — focused on get_diff filtering logic."""

import subprocess
import textwrap

import pytest

from purple.docker_runner import DockerRunner


# ---------------------------------------------------------------------------
# Pattern tests — validate the grep regexes built from the class constants
# without Docker.  We pipe sample file lists through the same grep pipeline
# that get_diff constructs.
# ---------------------------------------------------------------------------

def _build_filter_cmd() -> str:
    """Reproduce the grep pipeline that get_diff uses."""
    dir_pat = "|".join(
        d.replace(".", r"\.") for d in DockerRunner._DIFF_EXCLUDE_DIRS
    )
    ext_pat = "|".join(DockerRunner._DIFF_EXCLUDE_EXTS)
    substr_pat = "|".join(
        s.replace(".", r"\.") for s in DockerRunner._DIFF_EXCLUDE_SUBSTR
    )
    return (
        f"grep -v -E '(^|/)({dir_pat})(/)'"
        f" | grep -v -E '({substr_pat})'"
        f" | grep -v -E '\\.({ext_pat})$'"
    )


def _filter_files(paths: list[str]) -> list[str]:
    """Run the filter pipeline on a list of paths, return survivors."""
    if not paths:
        return []
    input_text = "\n".join(paths) + "\n"
    cmd = _build_filter_cmd()
    result = subprocess.run(
        ["bash", "-c", cmd],
        input=input_text,
        capture_output=True,
        text=True,
    )
    return [l for l in result.stdout.strip().splitlines() if l]


class TestDiffFilterPatterns:
    """Verify that the exclude patterns correctly classify file paths."""

    # -- Files that SHOULD survive the filter (source code) ----------------

    @pytest.mark.parametrize("path", [
        "src/user/email.js",
        "src/newfile.py",
        "tests/test_new.py",
        "lib/database/redis.go",
        "Makefile",
        "README.md",
        "setup.py",
        "build.go",               # "build" as filename, not directory
        "package.json",
        "src/venv_util.py",       # "venv" as substring in filename
    ])
    def test_source_files_pass(self, path):
        assert _filter_files([path]) == [path]

    # -- Files that SHOULD be filtered (runtime / binary) ------------------

    @pytest.mark.parametrize("path,reason", [
        ("appendonlydir/appendonly.aof.1.incr.aof", "redis AOF dir"),
        ("appendonlydir/foo.rdb", "redis dir"),
        ("node_modules/express/index.js", "node_modules dir"),
        ("src/__pycache__/foo.cpython-311.pyc", "__pycache__ dir"),
        (".tox/py39/lib/site.py", ".tox dir"),
        (".venv/lib/python3.11/os.py", ".venv dir"),
        (".mypy_cache/3.11/foo.py", ".mypy_cache dir"),
        (".pytest_cache/v/cache/foo", ".pytest_cache dir"),
        ("dump.rdb", "rdb extension"),
        ("data/test.sqlite3", "sqlite3 extension"),
        ("data/local.db", "db extension"),
        ("foo.pyc", "pyc extension"),
        ("lib/thing.so", "so extension"),
        ("output.log", "log extension"),
        ("server.pid", "pid extension"),
        ("icon.png", "png extension"),
        ("archive.tar.gz", "gz extension"),
        ("dist.zip", "zip extension"),
        ("my.egg-info/top_level.txt", "egg-info dir"),
    ])
    def test_junk_files_excluded(self, path, reason):
        assert _filter_files([path]) == [], f"expected {path} to be excluded ({reason})"

    # -- Batch test: mixed list --------------------------------------------

    def test_mixed_file_list(self):
        paths = [
            "src/user/email.js",
            "appendonlydir/appendonly.aof.1.incr.aof",
            "node_modules/foo/bar.js",
            "tests/test_new.py",
            "dump.rdb",
            ".tox/py39/lib/foo.py",
            "newfile.txt",
            "__pycache__/mod.pyc",
            "src/newmodule.py",
        ]
        result = _filter_files(paths)
        assert result == [
            "src/user/email.js",
            "tests/test_new.py",
            "newfile.txt",
            "src/newmodule.py",
        ]

    # -- Edge cases --------------------------------------------------------

    def test_empty_list(self):
        assert _filter_files([]) == []

    def test_all_junk(self):
        paths = ["dump.rdb", "node_modules/x.js", "foo.pyc"]
        assert _filter_files(paths) == []

    def test_all_source(self):
        paths = ["a.py", "b.js", "c.go"]
        assert _filter_files(paths) == paths

    def test_deeply_nested_excluded_dir(self):
        """A junk dir deep in the tree should still be caught."""
        assert _filter_files(["a/b/c/__pycache__/d.pyc"]) == []
        assert _filter_files(["a/b/node_modules/c/d.js"]) == []

    def test_dir_name_as_file_prefix_not_excluded(self):
        """Files whose name *starts with* an excluded dir shouldn't match."""
        # 'venv_helper.py' should pass — "venv" is only excluded as a directory
        assert _filter_files(["venv_helper.py"]) == ["venv_helper.py"]

    def test_extension_substring_not_excluded(self):
        """.db should not catch .dbs or .dbm."""
        assert _filter_files(["test.dbs"]) == ["test.dbs"]
        assert _filter_files(["shelf.dbm"]) == ["shelf.dbm"]


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestDiffExcludeConstants:
    """Verify the exclude-list constants are well-formed."""

    def test_no_leading_dots_in_exts(self):
        for ext in DockerRunner._DIFF_EXCLUDE_EXTS:
            assert not ext.startswith("."), f"Extension should not have leading dot: {ext}"

    def test_no_slashes_in_dirs(self):
        for d in DockerRunner._DIFF_EXCLUDE_DIRS:
            assert "/" not in d, f"Dir entry should not contain slash: {d}"

    def test_substr_entries_end_with_slash(self):
        for s in DockerRunner._DIFF_EXCLUDE_SUBSTR:
            assert s.endswith("/"), f"Substr entry should end with /: {s}"

    def test_no_duplicates_in_exts(self):
        assert len(DockerRunner._DIFF_EXCLUDE_EXTS) == len(set(DockerRunner._DIFF_EXCLUDE_EXTS))

    def test_no_duplicates_in_dirs(self):
        assert len(DockerRunner._DIFF_EXCLUDE_DIRS) == len(set(DockerRunner._DIFF_EXCLUDE_DIRS))


# ---------------------------------------------------------------------------
# Shell command construction test
# ---------------------------------------------------------------------------

class TestGetDiffCommand:
    """Verify the shell command get_diff builds is syntactically valid."""

    def test_command_parses_in_bash(self):
        """The shell command should be valid bash syntax (syntax-only check)."""
        dir_pat = "|".join(
            d.replace(".", r"\.") for d in DockerRunner._DIFF_EXCLUDE_DIRS
        )
        ext_pat = "|".join(DockerRunner._DIFF_EXCLUDE_EXTS)
        substr_pat = "|".join(
            s.replace(".", r"\.") for s in DockerRunner._DIFF_EXCLUDE_SUBSTR
        )
        patch_path = "/tmp/_patch.diff"
        cmd = (
            f"(git ls-files --others --exclude-standard"
            f" | grep -v -E '(^|/)({dir_pat})(/)'"
            f" | grep -v -E '({substr_pat})'"
            f" | grep -v -E '\\.({ext_pat})$'"
            f" | xargs -r -d '\\n' git add -N -- || true)"
            f" && git diff > {patch_path}"
            f" ; git reset 2>/dev/null || true"
        )
        result = subprocess.run(
            ["bash", "-n", "-c", cmd],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash syntax error: {result.stderr}"
