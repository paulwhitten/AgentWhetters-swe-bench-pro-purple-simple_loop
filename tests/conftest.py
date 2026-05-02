import socket
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://127.0.0.1:9022",
        help="Purple agent URL for integration tests (default: http://127.0.0.1:9022)",
    )


@pytest.fixture
def purple_url(request):
    url = request.config.getoption("--agent-url")
    # Skip integration tests when no server is reachable
    try:
        host = url.split("://", 1)[-1].split(":")[0]
        port = int(url.rsplit(":", 1)[-1].split("/")[0])
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError:
        pytest.skip(f"No agent server reachable at {url} (integration test)")
    return url
