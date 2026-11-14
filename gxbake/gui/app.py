"""Start the GxBake desktop app."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
	from PySide6.QtWidgets import QApplication

	from .main_window import MainWindow

	argv = sys.argv if argv is None else argv
	app = QApplication(argv)
	app.setApplicationName("GxBake")
	app.setOrganizationName("GxBake")
	app.setStyle("Fusion")
	win = MainWindow()
	paths = [a for a in argv[1:] if not a.startswith("-")]
	if paths:
		win.add_paths(paths)
	win.show()
	return app.exec()


if __name__ == "__main__":
	raise SystemExit(main())
