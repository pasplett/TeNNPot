"""Concise command-line reporting for invalid input and file errors."""

from contextlib import contextmanager


@contextmanager
def report_input_errors(parser, *, debug=False):
    """Exit on expected user errors, retaining tracebacks in debug mode."""
    try:
        yield
    except (ValueError, OSError) as error:
        if debug:
            raise
        parser.exit(2, f"{parser.prog}: error: {error}\n")
