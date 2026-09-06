"""Execute the standalone training CLI."""

from server.foundation.runtime_logging import configure_stderr_logging
from server.training_cli.cli import main

configure_stderr_logging()
main()
