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


def test_assimp_model_preview(tmp_path):
	from gxbake.core import models
	from gxbake.gui.main_window import MainWindow

	from . import fixtures as fx

	if not models.available():
		pytest.skip("assimp-py not installed")
	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	w = MainWindow()
	w.qsettings.clear()
	w.add_paths([str(fx.obj_cube(tmp_path / "src"))])
	item = w.tree.topLevelItem(0)
	item.setExpanded(True)
	assert item.childCount() == 2	# one row per material
	w.tree.setCurrentItem(item)
	end = time.time() + 10
	while time.time() < end and ("rendering" in w.preview_title.text() or w.inspector_form.rowCount() == 0):
		app.processEvents()
		time.sleep(0.01)
	assert w.preview_title.text() == "crate.obj"
	assert w.src_pane.canvas._image is not None and w.gx_pane.canvas._image is not None
	w.close()


def test_rose_character_preview(tmp_path):
	from gxbake.gui.main_window import MainWindow

	from .test_rose import _arm

	_arm(tmp_path)
	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	w = MainWindow()
	w.qsettings.clear()
	w.add_paths([str(tmp_path / "3DDATA")])
	assert w.tree.topLevelItemCount() == 1
	item = w.tree.topLevelItem(0)
	item.setExpanded(True)
	assert [item.child(i).text(0) for i in range(item.childCount())] == ["arm", "▶ stop", "▶ attack"]
	w.tree.setCurrentItem(item.child(2))
	end = time.time() + 10
	while time.time() < end and ("rendering" in w.preview_title.text() or w.inspector_form.rowCount() == 0):
		app.processEvents()
		time.sleep(0.01)
	assert w.preview_title.text().endswith("▶ attack")
	assert w.gx_pane.canvas._image is not None
	w.close()
