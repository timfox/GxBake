"""`python -m gxbake` (and the frozen executable): GUI by default, CLI with a subcommand."""

import sys

from gxbake.cli import main

if __name__ == "__main__":
	sys.exit(main())
