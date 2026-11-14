"""Offscreen smoke test: the window builds, loads a gamedir, previews and converts."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from .test_core import make_gamedir  # noqa: E402


def test_main_window_round_trip(tmp_path):
	from gxbake.gui.main_window import MainWindow

	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	w = MainWindow()
	w.qsettings.clear()
	w.out_edit.setText(str(tmp_path / "out"))
	w.add_paths([str(make_gamedir(tmp_path))])
	assert w.tree.topLevelItemCount() == 3	# skybox, bsp, wad

	def pump(cond, timeout=10):
		end = time.time() + timeout
		while time.time() < end and not cond():
			app.processEvents()
			time.sleep(0.01)
		return cond()

	w.tree.setCurrentItem(w.tree.topLevelItem(1))
	assert pump(lambda: "rendering" not in w.preview_title.text() and w.inspector_form.rowCount() > 0)
	w.convert(selected_only=False)
	assert pump(lambda: not w.busy, 30)
	assert (tmp_path / "out/media/wt_demo.gct").is_file()
	assert "main RAM" in w.budget.text(), (w.written, [a.outputs for a in w.project.assets], w.log_view.toPlainText()[-800:])
	w.close()
