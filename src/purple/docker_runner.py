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
        """Execute a shell command inside the container and return its output."""
        if not self._container:
            raise RuntimeError("Container not started")

        exec_result = self._container.exec_run(
            ["bash", "-c", cmd],
            workdir=REPO_DIR,
            demux=True,
        )
        stdout = (exec_result.output[0] or b"").decode(errors="replace") if exec_result.output else ""
        stderr = (exec_result.output[1] or b"").decode(errors="replace") if exec_result.output else ""
        combined = stdout
        if stderr:
            combined = combined + "\n" + stderr if combined else stderr
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

    def get_diff(self) -> str:
        """Return the current git diff in the container.

        Writes to a file first then extracts via Docker archive API to
        avoid exec_run demux truncation (docker-py can drop the last
        stdout frame when output doesn't end with a newline).
        """
        if not self._container:
            raise RuntimeError("Container not started")
        patch_path = "/tmp/_patch.diff"
        self.run(f"git diff > {patch_path}")
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
