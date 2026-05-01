"""
Docker runner — executes commands and reads files inside SWE-bench eval containers.

The green agent sends a Docker image URI per instance. The purple agent pulls that
image, starts a container, then explores the repo and generates patches inside it.
"""

from __future__ import annotations

import logging
import tarfile
import io
from dataclasses import dataclass

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)

REPO_DIR = "/app"
EXEC_TIMEOUT = 120  # seconds per command


@dataclass
class ExecResult:
    exit_code: int
    output: str


class DockerRunner:
    """Manage a throwaway container for a single SWE-bench instance."""

    def __init__(self, image_uri: str, base_commit: str, platform: str | None = None):
        self._image_uri = image_uri
        self._base_commit = base_commit
        self._platform = platform
        self._client: docker.DockerClient | None = None
        self._container: Container | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Pull the image and start a long-running container."""
        self._client = docker.from_env()
        try:
            pull_kwargs = {"platform": self._platform} if self._platform else {}
            self._client.images.pull(self._image_uri, **pull_kwargs)
        except Exception:
            try:
                self._client.images.get(self._image_uri)
                logger.info("Using locally cached image: %s", self._image_uri)
            except Exception as exc:
                raise RuntimeError(f"Cannot pull or find image {self._image_uri}: {exc}") from exc

        create_kwargs: dict = {
            "detach": True,
            "entrypoint": "/bin/bash",
            "command": ["-c", "tail -f /dev/null"],  # keep alive
            "working_dir": REPO_DIR,
        }
        if self._platform:
            create_kwargs["platform"] = self._platform

        self._container = self._client.containers.create(self._image_uri, **create_kwargs)
        self._container.start()
        logger.info("Started container %s from %s", self._container.short_id, self._image_uri)

        # Reset to base commit so the repo is in a clean state
        self.run(f"git checkout {self._base_commit} && git reset --hard {self._base_commit}")

    def stop(self) -> None:
        """Stop and remove the container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            logger.info("Removed container %s", self._container.short_id)
            self._container = None

    def cleanup_image(self) -> None:
        """Remove the pulled image to reclaim disk space."""
        if not self._client:
            return
        try:
            self._client.images.remove(self._image_uri, force=True)
            logger.info("Removed image: %s", self._image_uri)
        except Exception as exc:
            logger.warning("Failed to remove image %s: %s", self._image_uri, exc)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def run(self, cmd: str, timeout: int = EXEC_TIMEOUT) -> ExecResult:
        """Execute a shell command inside the container and return its output.

        docker-py's ``exec_run`` has no per-call timeout. To honor the
        ``timeout`` argument we wrap the command with coreutils ``timeout``
        so a hung command cannot consume the whole per-instance budget.
        On timeout, the wrapper exits with 124 (TERM) or 137 (KILL after
        grace period) and we annotate the output.
        """
        if not self._container:
            raise RuntimeError("Container not started")

        wrapped = ["timeout", "-k", "5", f"{timeout}s", "bash", "-c", cmd]
        exec_result = self._container.exec_run(
            wrapped,
            workdir=REPO_DIR,
            demux=True,
        )
        stdout = (exec_result.output[0] or b"").decode(errors="replace") if exec_result.output else ""
        stderr = (exec_result.output[1] or b"").decode(errors="replace") if exec_result.output else ""
        combined = stdout
        if stderr:
            combined = combined + "\n" + stderr if combined else stderr
        if exec_result.exit_code in (124, 137):
            note = f"\n[command timed out after {timeout}s]"
            combined = combined + note if combined else note.lstrip("\n")
        return ExecResult(exit_code=exec_result.exit_code, output=combined)

    def read_file(self, path: str, max_bytes: int = 100_000) -> str:
        """Read a file from the container."""
        result = self.run(f"head -c {max_bytes} {path}")
        if result.exit_code != 0:
            raise FileNotFoundError(f"Cannot read {path}: {result.output}")
        return result.output

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the container using a tar archive."""
        if not self._container:
            raise RuntimeError("Container not started")
        # Ensure parent directory exists
        parent = "/".join(path.split("/")[:-1])
        if parent:
            self.run(f"mkdir -p '{parent}'")
        # Preserve original file permissions if the file already exists
        stat_result = self.run(f"stat -c '%a' '{path}' 2>/dev/null")
        mode = int(stat_result.output.strip(), 8) if stat_result.exit_code == 0 else 0o644
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
        tar_buf.seek(0)
        self._container.put_archive(REPO_DIR, tar_buf)

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        """List directory tree inside the container."""
        result = self.run(f"find {path} -maxdepth {max_depth} -type f | head -500")
        return result.output

    def apply_patch(self, patch: str) -> ExecResult:
        """Write a patch file and apply it with git apply."""
        if not self._container:
            raise RuntimeError("Container not started")

        # Write patch via tar archive (works in Docker-outside-Docker)
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            data = patch.encode("utf-8")
            info = tarfile.TarInfo(name="fix.patch")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_buf.seek(0)
        self._container.put_archive(REPO_DIR, tar_buf)

        return self.run("git apply -v fix.patch && rm fix.patch")

    # Directories whose contents should never appear in a submitted patch.
    # Matched anywhere in the relative path (e.g. ``src/__pycache__/``).
    _DIFF_EXCLUDE_DIRS = (
        "appendonlydir", "node_modules", "__pycache__", ".tox",
        ".venv", "venv", ".eggs", "htmlcov",
        ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    )
    # Patterns matched as *substrings* anywhere in the path (for entries
    # like ``egg-info`` that appear as a suffix: ``my_pkg.egg-info/``).
    _DIFF_EXCLUDE_SUBSTR = (
        ".egg-info/",
    )
    # File extensions that indicate binary / runtime / build artifacts.
    # Stored without leading dot; the grep pattern adds ``\.`` prefix.
    _DIFF_EXCLUDE_EXTS = (
        "aof", "rdb", "db", "sqlite", "sqlite3",
        "pyc", "pyo", "o", "so", "dylib", "a",
        "class", "jar", "war", "whl", "egg",
        "log", "pid",
        "png", "jpg", "jpeg", "gif", "ico", "svg",
        "zip", "tar", "gz", "bz2", "xz",
    )

    def is_running(self) -> bool:
        """Check if the container is still running."""
        if not self._container:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            return False

    def get_diff(self) -> str:
        """Return the current git diff in the container.

        New (untracked) files are discovered via ``git ls-files --others``
        and selectively staged with ``git add -N`` so they appear in the
        diff.  Runtime artifacts (database files, compiled objects, package
        directories, media) are filtered out to keep the patch clean.

        Writes to a file first then extracts via Docker archive API to
        avoid exec_run demux truncation (docker-py can drop the last
        stdout frame when output doesn't end with a newline).
        """
        if not self._container:
            raise RuntimeError("Container not started")
        if not self.is_running():
            logger.error("Container %s is not running — cannot collect diff",
                         self._container.short_id)
            return ""
        patch_path = "/tmp/_patch.diff"

        # Build grep patterns to exclude junk directories and extensions.
        dir_pat = "|".join(d.replace(".", r"\.") for d in self._DIFF_EXCLUDE_DIRS)
        ext_pat = "|".join(self._DIFF_EXCLUDE_EXTS)
        substr_pat = "|".join(
            s.replace(".", r"\.") for s in self._DIFF_EXCLUDE_SUBSTR
        )

        # 1. List untracked files (respects .gitignore).
        # 2. Filter out runtime/data artifacts by directory and extension.
        # 3. Intent-to-add the remaining files so git diff includes them.
        # The subshell + || true ensures a zero exit even when grep filters
        # out every line (exit 1) or there are no untracked files at all.
        # After the diff is captured, git reset undoes the intent-to-add.
        # Use "git diff HEAD" to capture both staged and unstaged changes
        # relative to the last commit.  Plain "git diff" misses files that
        # the agent already staged with "git add".
        self.run(
            f"(git ls-files --others --exclude-standard"
            f" | grep -v -E '(^|/)({dir_pat})(/)'"
            f" | grep -v -E '({substr_pat})'"
            f" | grep -v -E '\\.({ext_pat})$'"  # noqa: ISC003
            f" | grep -v -F '.swe_baseline_test_output.txt'"
            f" | xargs -r -d '\\n' git add -N -- || true)"
            f" && git diff HEAD -- . ':!.swe_baseline_test_output.txt' > {patch_path}"
            f" ; git reset 2>/dev/null || true"
        )
        bits, _stat = self._container.get_archive(patch_path)
        raw = b"".join(bits)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            member = tar.getmembers()[0]
            data = tar.extractfile(member).read()
        self.run(f"rm -f {patch_path}")
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
