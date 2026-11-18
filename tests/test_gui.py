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


def test_panorama_sky_preview(tmp_path):
	from PIL import Image

	from gxbake.gui.main_window import MainWindow

	from .test_skybox import direction_panorama

	Image.fromarray(direction_panorama(1024)).save(tmp_path / "noon.png")
	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	w = MainWindow()
	w.qsettings.clear()
	w.out_edit.setText(str(tmp_path / "out"))
	w.add_paths([str(tmp_path / "noon.png")])
	item = w.tree.topLevelItem(0)
	assert item.text(1) == "Skybox"
	item.setExpanded(True)
	assert item.child(item.childCount() - 1).text(0) == "all sides"
	w.tree.setCurrentItem(item.child(item.childCount() - 1))
	end = time.time() + 20
	while time.time() < end and ("rendering" in w.preview_title.text() or w.inspector_form.rowCount() == 0):
		app.processEvents()
		time.sleep(0.01)
	assert w.preview_title.text() == "sky noon"
	assert w.gx_pane.canvas._image is not None
	w.convert(selected_only=False)
	end = time.time() + 30
	while time.time() < end and w.busy:
		app.processEvents()
		time.sleep(0.01)
	assert (tmp_path / "out/media/sky_noonft.cmp").stat().st_size == 8192
	w.close()


def test_extract_lumps_action(tmp_path):
	from gxbake.gui.main_window import MainWindow

	from . import lump_fixtures as lf

	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	gcd = tmp_path / "wd_demo.gcd"
	gcd.write_bytes(lf.world_dl(pages=2))
	w = MainWindow()
	w.qsettings.clear()
	w.out_edit.setText(str(tmp_path / "out"))
	w.add_paths([str(gcd)])
	w.tree.setCurrentItem(w.tree.topLevelItem(0))
	app.processEvents()
	assert w.a_lumps.isEnabled()
	w.extract_lumps()
	assert (tmp_path / "out/lumps/wd_demo_gcd/lightmaps/atlas.png").is_file(), w.log_view.toPlainText()
	w.close()


def test_video_settings_and_button():
	from gxbake.gui.main_window import FILE_FILTER, MainWindow

	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	w = MainWindow()
	w.qsettings.clear()
	panel = w.settings_panel
	panel.vformat.setCurrentText("thp")
	panel.thp_q.setValue(65)
	panel.thp_kbps.setValue(1500)
	s = w.project.settings
	assert (s.video_format, s.thp_quality, s.thp_max_kbps) == ("thp", 65, 1500.0)
	assert s.thp_options().max_kbps == 1500.0
	assert w.a_video.isEnabled()
	for group in FILE_FILTER.split(";;"):	# no glob twice in one filter
		globs = group[group.find("(") + 1:group.rfind(")")].split()
		assert len(globs) == len(set(globs)), group[:40]
	assert "*.thp" in FILE_FILTER and "*.bik" in FILE_FILTER
	w.close()
	app.processEvents()
