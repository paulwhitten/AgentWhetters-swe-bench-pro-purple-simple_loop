import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://127.0.0.1:9022",
        help="Purple agent URL for integration tests (default: http://127.0.0.1:9022)",
    )


@pytest.fixture
def purple_url(request):
    return request.config.getoption("--agent-url")
