"""GxBake main window: asset list, side-by-side GX preview, settings, export."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, QThreadPool, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
	QAbstractItemView,
	QCheckBox,
	QComboBox,
	QDockWidget,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QHeaderView,
	QLabel,
	QLineEdit,
	QListWidget,
	QMainWindow,
	QMessageBox,
	QPlainTextEdit,
	QProgressBar,
	QPushButton,
	QScrollArea,
	QSpinBox,
	QSplitter,
	QTabWidget,
	QToolBar,
	QTreeWidget,
	QTreeWidgetItem,
	QVBoxLayout,
	QWidget,
)

from .. import __version__
from ..core import audit, images, media, models, panda, rose
from ..core.project import CONVERTIBLE, Asset, Kind, Project, Settings
from .widgets import ImagePane, Task

ASSET_ROLE = Qt.ItemDataRole.UserRole
CHILD_ROLE = Qt.ItemDataRole.UserRole + 1
HAS_CHILDREN = {Kind.WAD, Kind.BSP, Kind.GCT, Kind.MDL, Kind.MODEL, Kind.SKYBOX}
_MODEL_GLOBS = " ".join(f"*{e}" for e in sorted(models.MODEL_EXTENSIONS | models.SOURCE2_MODEL_EXTENSIONS | models.DMX_MODEL_EXTENSIONS))
_IMAGE_GLOBS = " ".join(f"*{e}" for e in sorted(images.IMAGE_EXTENSIONS))
_AUDIO_GLOBS = " ".join(f"*{e}" for e in sorted(media.AUDIO_EXTENSIONS))
_ROSE_GLOBS = " ".join(f"*{e}" for e in sorted(rose.ROSE_EXTENSIONS))
_USD_GLOBS = " ".join(f"*{e}" for e in sorted(models.USD_EXTENSIONS))
FILE_FILTER = (
	f"Source assets ({_IMAGE_GLOBS} *.wad *.bsp *.vmap_c *.vwrld_c *.mdl *.spr *.sp2 *.pcf *.vpcf_c *.avi *.mp4 *.mkv *.mov *.webm {_AUDIO_GLOBS} "
	f"Source assets ({_IMAGE_GLOBS} *.wad *.bsp *.mdl *.spr *.sp2 *.avi *.mp4 *.mkv *.mov *.webm {_AUDIO_GLOBS} "
	f"{_MODEL_GLOBS} {_ROSE_GLOBS} main_file_cache.dat2 main_file_cache.dat *.dat *.ffximdl *.spw *.bgw *.hrc *.p *.ff7mdl);;"
	f"{_MODEL_GLOBS} {_USD_GLOBS} {_ROSE_GLOBS});;"
_PANDA_GLOBS = "*.egg *.bam *.pz *.pandactor"
FILE_FILTER = (
	f"Source assets ({_IMAGE_GLOBS} *.wad *.bsp *.mdl *.spr *.sp2 *.avi *.mp4 *.mkv *.mov *.webm {_AUDIO_GLOBS} "
	f"{_MODEL_GLOBS} {_ROSE_GLOBS} {_PANDA_GLOBS});;"
	f"3D models via Assimp ({_MODEL_GLOBS});;"
	f"USD scenes via FreeUSD ({_USD_GLOBS});;"
	"Source, Quake and Quake II (*.mdl *.vtf *.bsp *.wad *.spr *.sp2 *.wal);;"
	f"ROSE Online models ({_ROSE_GLOBS});;"
	f"Panda3D and Toontown models ({_PANDA_GLOBS});;"
	f"{_MODEL_GLOBS} {_ROSE_GLOBS} main_file_cache.dat2 main_file_cache.dat);;"
	f"3D models via Assimp ({_MODEL_GLOBS});;"
	"Source, Quake and Quake II (*.mdl *.vtf *.bsp *.wad *.spr *.sp2 *.wal);;"
	f"ROSE Online models ({_ROSE_GLOBS});;"
	"RuneScape caches (main_file_cache.dat2 main_file_cache.dat);;"
	"Final Fantasy XI (*.dat *.ffximdl *.spw *.bgw);;"
	"Final Fantasy VII field models (*.hrc *.p *.ff7mdl);;"
	"Final Fantasy VII battle skeletons (*aa);;"
	"GameCube outputs (*.gct *.cmp *.gcvid *.gcpcm);;All files (*)"
)


class SettingsPanel(QWidget):
	"""Optimisation settings, bound to a Settings object."""

	def __init__(self, settings: Settings, on_change, parent=None):
		super().__init__(parent)
		self.s = settings
		self.on_change = on_change
		outer = QVBoxLayout(self)

		tex = QGroupBox("Textures (CMPR)")
		f = QFormLayout(tex)
		self.max_edge = self._combo(["64", "128", "256", "512", "1024"], str(settings.image.max_edge))
		self.max_edge.setToolTip("Long side clamp for loose images. Sides are rounded up to powers of two (GX_REPEAT needs them).")
		f.addRow("Max size", self.max_edge)
		self.mipmaps = QCheckBox("Generate mip chain")
		self.mipmaps.setChecked(settings.image.mipmaps)
		f.addRow(self.mipmaps)
		self.alpha = self._combo(["auto", "on", "off"], settings.image.alpha)
		self.alpha.setToolTip("CMPR stores 1-bit alpha. 'auto' enables it when the image has transparent texels.")
		f.addRow("1-bit alpha", self.alpha)
		self.threshold = self._spin(1, 255, settings.image.alpha_threshold)
		f.addRow("Alpha cutoff", self.threshold)
		self.resample = self._combo(["lanczos", "bicubic", "nearest"], settings.image.resample)
		f.addRow("Resize filter", self.resample)
		self.pack = QCheckBox("Pack loose images into one .gct")
		self.pack.setChecked(settings.pack_images)
		f.addRow(self.pack)
		self.pack_name = QLineEdit(settings.pack_name)
		f.addRow("Pack name", self.pack_name)
		self.raw_cmp = QCheckBox("Also write raw level-0 .cmp")
		self.raw_cmp.setChecked(settings.raw_cmp)
		f.addRow(self.raw_cmp)
		self.sky = self._combo(["64", "128", "256"], str(settings.sky_size))
		self.sky.setToolTip("The engine expects 128x128 sky sides (8 KiB each).")
		f.addRow("Sky side size", self.sky)
		outer.addWidget(tex)

		sky = QGroupBox("Skies from panoramas")
		f = QFormLayout(sky)
		self.sky_detect = QCheckBox("Treat panoramas, crosses, strips and .hdr as skies")
		self.sky_detect.setToolTip("2:1 images 1024+ wide, 6:1 strips, 4:3/3:4 crosses with empty corners and .hdr files "
			"become skyboxes when added. Use Add Skybox… for anything else.")
		self.sky_detect.setChecked(settings.detect_panoramas)
		f.addRow(self.sky_detect)
		self.sky_yaw = QDoubleSpinBox()
		self.sky_yaw.setRange(0.0, 359.0)
		self.sky_yaw.setDecimals(0)
		self.sky_yaw.setSingleStep(90.0)
		self.sky_yaw.setWrapping(True)
		self.sky_yaw.setValue(settings.sky_yaw % 360)
		self.sky_yaw.setToolTip("Engine yaw the panorama centre (or the cross's front face) looks along: "
			"0 = rt (+X), 90 = bk (+Y), 180 = lf (-X), 270 = ft (-Y).")
		f.addRow("Centre faces yaw", self.sky_yaw)
		self.sky_fov = QDoubleSpinBox()
		self.sky_fov.setRange(30.0, 360.0)
		self.sky_fov.setDecimals(0)
		self.sky_fov.setValue(settings.sky_fov)
		self.sky_fov.setToolTip("A flat photo made into a sky covers this many degrees; mirrored copies fill the rest.")
		f.addRow("Photo width (°)", self.sky_fov)
		self.sky_exposure = QDoubleSpinBox()
		self.sky_exposure.setRange(-8.0, 8.0)
		self.sky_exposure.setSingleStep(0.5)
		self.sky_exposure.setValue(settings.sky_exposure)
		self.sky_exposure.setToolTip(".hdr panoramas are tone-mapped with automatic exposure; this adds stops.")
		f.addRow("HDR exposure (stops)", self.sky_exposure)
		outer.addWidget(sky)

		mdl = QGroupBox("Models and sprites")
		f = QFormLayout(mdl)
		self.lean_mesh = QCheckBox("Lean mesh (strip skins) → gc_studio/<name>.mdl")
		self.lean_mesh.setChecked(settings.mdl_lean_mesh)
		f.addRow(self.lean_mesh)
		self.skin_comp = QCheckBox("Skin companion → gc_studio/<name>t.mdl")
		self.skin_comp.setChecked(settings.mdl_skin_companion)
		f.addRow(self.skin_comp)
		self.mdl_edge = self._combo(["32", "64", "128", "256"], str(settings.mdl_max_edge))
		f.addRow("Skin max edge", self.mdl_edge)
		self.spr_scale = self._combo(["1", "2", "4"], str(settings.spr_scale))
		self.spr_scale.setToolTip("Sprite downsample factor; alpha-test sheets keep thin strokes.")
		f.addRow("Sprite scale ÷", self.spr_scale)
		outer.addWidget(mdl)

		imp = QGroupBox("3D model import (Assimp, USD, Source, Quake)")
		f = QFormLayout(imp)
		if not models.available():
			warn = QLabel("assimp-py is not installed, so OBJ/FBX/glTF files can't be read.\npip install assimp-py")
			warn.setStyleSheet("color: #d9a63c;")
			f.addRow(warn)
		elif not models.skeletal_supported():
			warn = QLabel("This assimp-py has no bone API (needs 1.2+ on Python 3.12+);\nmodels import static.")
			warn.setStyleSheet("color: #d9a63c;")
			f.addRow(warn)
		from ..core import usd as _usd

		if not _usd.available():
			warn = QLabel("FreeUSD is not installed, so .usd/.usda/.usdz files can't be read.\npip install 'gxbake[usd]'")
			warn.setStyleSheet("color: #d9a63c;")
			f.addRow(warn)
		self.model_scale = QDoubleSpinBox()
		self.model_scale.setRange(0.001, 10000.0)
		self.model_scale.setDecimals(3)
		self.model_scale.setValue(settings.model_scale)
		self.model_scale.setToolTip("Source units → GoldSrc units (about an inch each). glTF/FBX in metres: try 39.37.")
		f.addRow("Scale", self.model_scale)
		self.model_up = self._combo(["y", "z"], settings.model_up_axis)
		self.model_up.setToolTip("y: Y-up source (glTF, OBJ, Assimp's FBX), +Z forward becomes the model's +X forward.\nz: already Z-up like GoldSrc.")
		f.addRow("Source up axis", self.model_up)
		self.model_skin = self._combo(["64", "128", "256", "512"], str(settings.model_skin_max_edge))
		self.model_skin.setToolTip("Skin size in the full models/<name>.mdl. The gc_studio companion uses 'Skin max edge' above.")
		f.addRow("Import skin max", self.model_skin)
		self.model_full = QCheckBox("Also write full studio model → models/<name>.mdl")
		self.model_full.setChecked(settings.model_write_full)
		f.addRow(self.model_full)
		self.model_static = QCheckBox("Bake static (ignore skeleton and animations)")
		self.model_static.setChecked(settings.model_static)
		f.addRow(self.model_static)
		self.model_fps = self._spin(1, 60, settings.model_anim_fps)
		self.model_fps.setToolTip("Sample rate for baked sequences. Lower is smaller; GoldSrc models are usually 10-30.")
		f.addRow("Animation fps", self.model_fps)
		self.model_clips = QCheckBox("Merge clip files (<name>@<clip>.fbx/.bvh…)")
		self.model_clips.setChecked(settings.model_clip_files)
		f.addRow(self.model_clips)
		self.model_blend = QCheckBox("Helper bones for blended weights (Source models)")
		self.model_blend.setToolTip("Adds rigid helper bones that follow two-bone blends, so joints bend instead of tearing.\n"
			"Costs a larger animation block; untick to snap each vertex to its heaviest bone.")
		self.model_blend.setChecked(settings.model_blend_bones)
		f.addRow(self.model_blend)
		outer.addWidget(imp)

		ro = QGroupBox("ROSE Online models (.zms .zsc .chr .rosemdl)")
		f = QFormLayout(ro)
		self.rose_scale = QDoubleSpinBox()
		self.rose_scale.setRange(0.01, 1000.0)
		self.rose_scale.setDecimals(2)
		self.rose_scale.setValue(settings.rose_scale)
		self.rose_scale.setToolTip("GoldSrc units per metre. 39.37 keeps real size (a ROSE avatar is about 55 units tall).")
		f.addRow("Units per metre", self.rose_scale)
		self.rose_yaw = QDoubleSpinBox()
		self.rose_yaw.setRange(-180.0, 180.0)
		self.rose_yaw.setDecimals(0)
		self.rose_yaw.setValue(settings.rose_yaw)
		self.rose_yaw.setToolTip("Turn about the up axis. ROSE models face -Y; 90 makes them face GoldSrc's +X.")
		f.addRow("Turn (degrees)", self.rose_yaw)
		outer.addWidget(ro)

		pa = QGroupBox("Panda3D and Toontown models (.egg .bam .pandactor)")
		f = QFormLayout(pa)
		self.panda_scale = QDoubleSpinBox()
		self.panda_scale.setRange(0.01, 1000.0)
		self.panda_scale.setDecimals(2)
		self.panda_scale.setValue(settings.panda_scale)
		self.panda_scale.setToolTip("GoldSrc units per Panda unit. Toontown's unit is a foot, so 12 keeps GoldSrc's inch scale.")
		f.addRow("Units per Panda unit", self.panda_scale)
		self.panda_yaw = QDoubleSpinBox()
		self.panda_yaw.setRange(-180.0, 180.0)
		self.panda_yaw.setDecimals(0)
		self.panda_yaw.setValue(settings.panda_yaw)
		self.panda_yaw.setToolTip("Turn about the up axis. Panda models face +Y; -90 makes them face GoldSrc's +X.")
		f.addRow("Turn (degrees)", self.panda_yaw)
		if not panda.bam_available():
			f.addRow(QLabel(".bam needs Panda3D (pip install panda3d) or bam2egg on PATH; .egg works as is."))
		outer.addWidget(pa)
		rsg = QGroupBox("RuneScape caches (main_file_cache.dat2 / .dat)")
		f = QFormLayout(rsg)
		self.rs_scale = QDoubleSpinBox()
		self.rs_scale.setRange(0.01, 10.0)
		self.rs_scale.setDecimals(3)
		self.rs_scale.setSingleStep(0.05)
		self.rs_scale.setValue(settings.rs_scale)
		self.rs_scale.setToolTip("GoldSrc units per model unit (128 units are one tile). 0.35 makes a player about 70 units tall.")
		f.addRow("Units per model unit", self.rs_scale)
		self.rs_yaw = QDoubleSpinBox()
		self.rs_yaw.setRange(-180.0, 180.0)
		self.rs_yaw.setDecimals(0)
		self.rs_yaw.setValue(settings.rs_yaw)
		self.rs_yaw.setToolTip("Extra turn about the up axis. Models are already turned to face GoldSrc's +X.")
		f.addRow("Turn (degrees)", self.rs_yaw)
		outer.addWidget(rsg)

		xi = QGroupBox("Final Fantasy XI (ROM .dat, .ffximdl, .spw/.bgw)")
		f = QFormLayout(xi)
		self.ffxi_scale = QDoubleSpinBox()
		self.ffxi_scale.setRange(0.01, 1000.0)
		self.ffxi_scale.setDecimals(2)
		self.ffxi_scale.setValue(settings.ffxi_scale)
		self.ffxi_scale.setToolTip("GoldSrc units per FFXI unit (taken as a metre; not yet checked against real data).")
		f.addRow("Units per FFXI unit", self.ffxi_scale)
		self.ffxi_yaw = QDoubleSpinBox()
		self.ffxi_yaw.setRange(-180.0, 180.0)
		self.ffxi_yaw.setDecimals(0)
		self.ffxi_yaw.setValue(settings.ffxi_yaw)
		self.ffxi_yaw.setToolTip("Extra turn about the up axis. Models are turned from FFXI's y-down space and assumed to face +X.")
		f.addRow("Turn (degrees)", self.ffxi_yaw)
		outer.addWidget(xi)

		f7 = QGroupBox("Final Fantasy VII field and battle models (.hrc, .p, .ff7mdl, <base>aa)")
		f = QFormLayout(f7)
		self.ff7_scale = QDoubleSpinBox()
		self.ff7_scale.setRange(0.01, 1000.0)
		self.ff7_scale.setDecimals(2)
		self.ff7_scale.setValue(settings.ff7_scale)
		self.ff7_scale.setToolTip("GoldSrc units per FF7 field unit. 2.5 makes a field character about 70 units tall (unverified).")
		f.addRow("Units per FF7 unit", self.ff7_scale)
		self.ff7_battle_scale = QDoubleSpinBox()
		self.ff7_battle_scale.setRange(0.0, 100.0)
		self.ff7_battle_scale.setDecimals(4)
		self.ff7_battle_scale.setValue(settings.ff7_battle_scale)
		self.ff7_battle_scale.setSpecialValueText("auto (72 units tall)")
		self.ff7_battle_scale.setToolTip("Battle models: GoldSrc units per FF7 battle unit. Auto fits each model to 72 units tall.")
		f.addRow("Battle units per FF7 unit", self.ff7_battle_scale)
		self.ff7_yaw = QDoubleSpinBox()
		self.ff7_yaw.setRange(-180.0, 180.0)
		self.ff7_yaw.setDecimals(0)
		self.ff7_yaw.setValue(settings.ff7_yaw)
		self.ff7_yaw.setToolTip("Extra turn about the up axis. Models are turned from FF7's y-down space and assumed to face -Z.")
		f.addRow("Turn (degrees)", self.ff7_yaw)
		outer.addWidget(f7)

		vid = QGroupBox("Video and audio")
		f = QFormLayout(vid)
		self.vw = self._spin(8, 640, settings.video_width, step=8)
		self.vh = self._spin(8, 480, settings.video_height, step=8)
		row = QHBoxLayout()
		row.addWidget(self.vw)
		row.addWidget(QLabel("×"))
		row.addWidget(self.vh)
		f.addRow("GCVID size", row)
		self.fps = self._spin(1, 60, settings.video_fps)
		f.addRow("Frame rate", self.fps)
		self.vaudio = QCheckBox("Write .gcpcm audio track")
		self.vaudio.setChecked(settings.video_audio)
		f.addRow(self.vaudio)
		outer.addWidget(vid)

		maps = QGroupBox("Maps (WAD search)")
		v = QVBoxLayout(maps)
		v.addWidget(QLabel("Extra folders searched for WADs when a BSP\nreferences external textures:"))
		self.wad_list = QListWidget()
		self.wad_list.addItems(settings.wad_dirs)
		self.wad_list.setMaximumHeight(80)
		v.addWidget(self.wad_list)
		row = QHBoxLayout()
		add = QPushButton("Add…")
		rem = QPushButton("Remove")
		add.clicked.connect(self._add_wad_dir)
		rem.clicked.connect(self._remove_wad_dir)
		row.addWidget(add)
		row.addWidget(rem)
		v.addLayout(row)
		row = QHBoxLayout()
		self.cache = QLineEdit(settings.cache_dir)
		self.cache.setPlaceholderText("Encode cache folder (optional)")
		pick = QPushButton("…")
		pick.setMaximumWidth(30)
		pick.clicked.connect(self._pick_cache)
		row.addWidget(self.cache)
		row.addWidget(pick)
		v.addLayout(row)
		outer.addWidget(maps)
		outer.addStretch(1)

		for w in (self.max_edge, self.alpha, self.resample, self.sky, self.mdl_edge, self.spr_scale, self.model_up, self.model_skin):
			w.currentTextChanged.connect(self._sync)
		for w in (self.mipmaps, self.pack, self.raw_cmp, self.lean_mesh, self.skin_comp, self.vaudio, self.model_full,
				self.model_static, self.model_clips, self.model_blend, self.sky_detect):
			w.toggled.connect(self._sync)
		for w in (self.threshold, self.vw, self.vh, self.fps, self.model_scale, self.model_fps, self.rose_scale, self.rose_yaw,
				self.rs_scale, self.rs_yaw, self.ffxi_scale, self.ffxi_yaw, self.ff7_scale, self.ff7_battle_scale, self.ff7_yaw, self.sky_yaw, self.sky_fov, self.sky_exposure):
				self.panda_scale, self.panda_yaw,
				self.sky_yaw, self.sky_fov, self.sky_exposure):
				self.rs_scale, self.rs_yaw, self.sky_yaw, self.sky_fov, self.sky_exposure):
			w.valueChanged.connect(self._sync)
		self.pack_name.editingFinished.connect(self._sync)
		self.cache.editingFinished.connect(self._sync)

	def _combo(self, items, current):
		c = QComboBox()
		c.addItems(items)
		c.setCurrentText(current)
		return c

	def _spin(self, lo, hi, val, step=1):
		s = QSpinBox()
		s.setRange(lo, hi)
		s.setSingleStep(step)
		s.setValue(val)
		return s

	def _add_wad_dir(self):
		d = QFileDialog.getExistingDirectory(self, "Folder with WAD files")
		if d:
			self.wad_list.addItem(d)
			self._sync()

	def _remove_wad_dir(self):
		for item in self.wad_list.selectedItems():
			self.wad_list.takeItem(self.wad_list.row(item))
		self._sync()

	def _pick_cache(self):
		d = QFileDialog.getExistingDirectory(self, "Encode cache folder")
		if d:
			self.cache.setText(d)
			self._sync()

	def _sync(self, *_):
		s = self.s
		s.image.max_edge = int(self.max_edge.currentText())
		s.image.mipmaps = self.mipmaps.isChecked()
		s.image.alpha = self.alpha.currentText()
		s.image.alpha_threshold = self.threshold.value()
		s.image.resample = self.resample.currentText()
		s.pack_images = self.pack.isChecked()
		s.pack_name = self.pack_name.text().strip() or "textures"
		s.raw_cmp = self.raw_cmp.isChecked()
		s.sky_size = int(self.sky.currentText())
		s.detect_panoramas = self.sky_detect.isChecked()
		s.sky_yaw = self.sky_yaw.value()
		s.sky_fov = self.sky_fov.value()
		s.sky_exposure = self.sky_exposure.value()
		s.mdl_lean_mesh = self.lean_mesh.isChecked()
		s.mdl_skin_companion = self.skin_comp.isChecked()
		s.mdl_max_edge = int(self.mdl_edge.currentText())
		s.spr_scale = int(self.spr_scale.currentText())
		s.model_scale = self.model_scale.value()
		s.model_up_axis = self.model_up.currentText()
		s.model_skin_max_edge = int(self.model_skin.currentText())
		s.model_write_full = self.model_full.isChecked()
		s.model_static = self.model_static.isChecked()
		s.model_anim_fps = self.model_fps.value()
		s.model_clip_files = self.model_clips.isChecked()
		s.model_blend_bones = self.model_blend.isChecked()
		s.rose_scale = self.rose_scale.value()
		s.rose_yaw = self.rose_yaw.value()
		s.panda_scale = self.panda_scale.value()
		s.panda_yaw = self.panda_yaw.value()
		s.rs_scale = self.rs_scale.value()
		s.rs_yaw = self.rs_yaw.value()
		s.ffxi_scale = self.ffxi_scale.value()
		s.ffxi_yaw = self.ffxi_yaw.value()
		s.ff7_scale = self.ff7_scale.value()
		s.ff7_yaw = self.ff7_yaw.value()
		s.ff7_battle_scale = self.ff7_battle_scale.value()
		s.video_width = self.vw.value()
		s.video_height = self.vh.value()
		s.video_fps = self.fps.value()
		s.video_audio = self.vaudio.isChecked()
		s.wad_dirs = [self.wad_list.item(i).text() for i in range(self.wad_list.count())]
		s.cache_dir = self.cache.text().strip()
		self.on_change()


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle(f"GxBake {__version__} — GameCube asset baker")
		self.resize(1400, 860)
		self.setAcceptDrops(True)
		self.qsettings = QSettings("GxBake", "GxBake")
		self.project = Project(self._load_settings())
		self.pool = QThreadPool.globalInstance()
		self.preview_gen = 0
		self.written: dict[str, int] = {}	# output rel path -> bytes, across runs
		self.busy = False
		self._preview_timer = QTimer(self, singleShot=True, interval=150)
		self._preview_timer.timeout.connect(self.refresh_preview)

		self._build_toolbar()
		self._build_body()
		self._build_log()
		self._build_status()
		self._update_buttons()

	# -- layout ----------------------------------------------------------------

	def _build_toolbar(self):
		tb = QToolBar("Main")
		tb.setMovable(False)
		self.addToolBar(tb)

		def act(text, slot, shortcut=None, tip=None):
			a = QAction(text, self)
			a.triggered.connect(slot)
			if shortcut:
				a.setShortcut(QKeySequence(shortcut))
			if tip:
				a.setToolTip(tip)
			tb.addAction(a)
			return a

		self.a_add = act("Add Files…", self.add_files, "Ctrl+O")
		self.a_add_dir = act("Add Folder…", self.add_folder, "Ctrl+Shift+O", "Scan a gamedir for convertible assets")
		self.a_add_sky = act("Add Skybox…", self.add_skybox, None,
			"A panorama, cubemap cross or strip, photo, or six face images → the six 128² CMPR sky sides")
		self.a_remove = act("Remove", self.remove_selected, "Del")
		self.a_clear = act("Clear", self.clear_assets)
		tb.addSeparator()
		self.a_convert_sel = act("Convert Selected", lambda: self.convert(selected_only=True), "Ctrl+B")
		self.a_convert_all = act("Convert All", lambda: self.convert(selected_only=False), "Ctrl+Shift+B")
		tb.addSeparator()
		tb.addWidget(QLabel(" Output: "))
		self.out_edit = QLineEdit(self.qsettings.value("output_dir", str(Path.home() / "gxbake-out")))
		self.out_edit.setMinimumWidth(320)
		tb.addWidget(self.out_edit)
		act("Browse…", self.pick_output)
		act("Open Output", self.open_output)

	def _build_body(self):
		split = QSplitter(Qt.Orientation.Horizontal)

		self.tree = QTreeWidget()
		self.tree.setHeaderLabels(["Asset", "Type", "Source", "GameCube", "Status"])
		self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
		header = self.tree.header()
		header.setStretchLastSection(True)
		header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
		self.tree.setColumnWidth(0, 170)
		for col in range(1, 4):
			header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
		self.tree.itemExpanded.connect(self._populate_children)
		self.tree.currentItemChanged.connect(lambda *_: self._preview_timer.start())
		self.tree.itemSelectionChanged.connect(self._update_buttons)
		split.addWidget(self.tree)

		middle = QWidget()
		mv = QVBoxLayout(middle)
		mv.setContentsMargins(0, 0, 0, 0)
		bar = QHBoxLayout()
		self.preview_title = QLabel("Drop files or a gamedir here to begin.")
		self.preview_title.setStyleSheet("font-size: 14px; font-weight: bold;")
		bar.addWidget(self.preview_title, 1)
		bar.addWidget(QLabel("Zoom"))
		self.zoom = QComboBox()
		self.zoom.addItems(["Fit", "1×", "2×", "4×", "8×"])
		self.zoom.currentTextChanged.connect(self._apply_zoom)
		bar.addWidget(self.zoom)
		mv.addLayout(bar)
		panes = QSplitter(Qt.Orientation.Horizontal)
		self.src_pane = ImagePane("Source")
		self.gx_pane = ImagePane("GameCube (decoded)")
		panes.addWidget(self.src_pane)
		panes.addWidget(self.gx_pane)
		panes.setSizes([1, 1])
		panes.setStretchFactor(0, 1)
		panes.setStretchFactor(1, 1)
		mv.addWidget(panes, 1)
		split.addWidget(middle)

		tabs = QTabWidget()
		self.inspector = QWidget()
		self.inspector_form = QFormLayout(self.inspector)
		insp_scroll = QScrollArea()
		insp_scroll.setWidget(self.inspector)
		insp_scroll.setWidgetResizable(True)
		tabs.addTab(insp_scroll, "Inspector")
		self.settings_panel = SettingsPanel(self.project.settings, self._settings_changed)
		set_scroll = QScrollArea()
		set_scroll.setWidget(self.settings_panel)
		set_scroll.setWidgetResizable(True)
		tabs.addTab(set_scroll, "Settings")
		split.addWidget(tabs)

		split.setSizes([460, 680, 340])
		self.setCentralWidget(split)

	def _build_log(self):
		dock = QDockWidget("Log", self)
		dock.setObjectName("log")
		self.log_view = QPlainTextEdit()
		self.log_view.setReadOnly(True)
		self.log_view.setMaximumBlockCount(5000)
		self.log_view.setStyleSheet("font-family: monospace;")
		dock.setWidget(self.log_view)
		self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
		self.resizeDocks([dock], [140], Qt.Orientation.Vertical)

	def _build_status(self):
		self.progress = QProgressBar()
		self.progress.setMaximumWidth(260)
		self.progress.setVisible(False)
		self.budget = QLabel("")
		self.statusBar().addPermanentWidget(self.budget)
		self.statusBar().addPermanentWidget(self.progress)

	# -- settings persistence -------------------------------------------------

	def _load_settings(self) -> Settings:
		raw = self.qsettings.value("settings")
		if raw:
			try:
				return Settings.from_json(json.loads(raw))
			except (TypeError, ValueError):
				pass
		return Settings()

	def _settings_changed(self):
		self.qsettings.setValue("settings", json.dumps(self.project.settings.to_json()))
		self._preview_timer.start()

	def closeEvent(self, event):	# noqa: N802
		self.qsettings.setValue("output_dir", self.out_edit.text())
		super().closeEvent(event)

	# -- assets ----------------------------------------------------------------

	def log(self, msg: str):
		self.log_view.appendPlainText(msg)

	def add_paths(self, paths):
		added = []
		for p in paths:
			added += self.project.add_path(Path(p))
		for asset in added:
			self._add_item(asset)
		if added:
			self.log(f"Added {len(added)} asset(s).")
			if self.tree.currentItem() is None:
				self.tree.setCurrentItem(self.tree.topLevelItem(0))
		self._update_buttons()
		self._update_budget()

	def _add_item(self, asset: Asset):
		item = QTreeWidgetItem()
		item.setData(0, ASSET_ROLE, asset)
		self._fill_row(item, asset)
		if asset.kind in HAS_CHILDREN:
			item.addChild(QTreeWidgetItem(["…"]))
		self.tree.addTopLevelItem(item)

	def _fill_row(self, item: QTreeWidgetItem, asset: Asset):
		item.setText(0, asset.name)
		item.setText(1, asset.kind.value)
		item.setText(2, audit.human_bytes(asset.size))
		item.setText(3, audit.human_bytes(asset.output_size) if asset.outputs else "")
		item.setText(4, asset.status)
		tip = "\n".join(asset.warnings)
		item.setToolTip(0, tip or str(asset.path))
		color = None
		if asset.status == "Failed":
			color = QColor(220, 80, 80)
		elif asset.warnings:
			color = QColor(220, 170, 60)
		elif asset.kind not in CONVERTIBLE:
			color = QColor(140, 140, 140)
		for col in range(5):
			if color:
				item.setForeground(col, color)
			else:
				item.setData(col, Qt.ItemDataRole.ForegroundRole, None)

	def _populate_children(self, item: QTreeWidgetItem):
		asset = item.data(0, ASSET_ROLE)
		if asset is None or item.childCount() != 1 or item.child(0).text(0) != "…":
			return
		item.takeChildren()
		for name in self.project.children(asset):
			child = QTreeWidgetItem([name])
			child.setData(0, ASSET_ROLE, asset)
			child.setData(0, CHILD_ROLE, name)
			item.addChild(child)
		self._fill_row(item, asset)

	def _selected_assets(self) -> list[Asset]:
		seen, out = set(), []
		for item in self.tree.selectedItems():
			a = item.data(0, ASSET_ROLE)
			if a is not None and id(a) not in seen:
				seen.add(id(a))
				out.append(a)
		return out

	def add_files(self):
		files, _ = QFileDialog.getOpenFileNames(self, "Add assets", self.qsettings.value("last_dir", ""), FILE_FILTER)
		if files:
			self.qsettings.setValue("last_dir", str(Path(files[0]).parent))
			self.add_paths(files)

	def add_skybox(self):
		files, _ = QFileDialog.getOpenFileNames(self, "Panorama, cross, strip, photo, or six faces",
			self.qsettings.value("last_dir", ""), f"Images ({_IMAGE_GLOBS});;All files (*)")
		if not files:
			return
		self.qsettings.setValue("last_dir", str(Path(files[0]).parent))
		if len(files) not in (1, 6):
			QMessageBox.warning(self, "Add Skybox", "Pick one image, or the six face images.")
			return
		try:
			added = self.project.add_skybox([Path(f) for f in files], "faces" if len(files) == 6 else "auto")
		except (OSError, ValueError) as exc:
			QMessageBox.warning(self, "Add Skybox", str(exc))
			return
		for asset in added:
			self._add_item(asset)
		if added:
			self.log(f"Added sky {added[0].extra['name']}.")
			self.tree.setCurrentItem(self.tree.topLevelItem(self.tree.topLevelItemCount() - 1))
		self._update_buttons()
		self._update_budget()

	def add_folder(self):
		d = QFileDialog.getExistingDirectory(self, "Add folder (scanned recursively)", self.qsettings.value("last_dir", ""))
		if d:
			self.qsettings.setValue("last_dir", d)
			self.add_paths([d])

	def remove_selected(self):
		if self.busy:
			return
		for asset in self._selected_assets():
			self.project.remove(asset)
		for i in reversed(range(self.tree.topLevelItemCount())):
			if self.tree.topLevelItem(i).data(0, ASSET_ROLE) not in self.project.assets:
				self.tree.takeTopLevelItem(i)
		self._update_buttons()
		self._update_budget()

	def clear_assets(self):
		if self.busy:
			return
		self.project.assets.clear()
		self.tree.clear()
		self._show_preview(None)
		self._update_buttons()
		self._update_budget()

	def dragEnterEvent(self, e):	# noqa: N802
		if e.mimeData().hasUrls():
			e.acceptProposedAction()

	def dropEvent(self, e):	# noqa: N802
		paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
		self.add_paths(paths)

	# -- preview ---------------------------------------------------------------

	def refresh_preview(self):
		item = self.tree.currentItem()
		if item is None:
			self._show_preview(None)
			return
		asset = item.data(0, ASSET_ROLE)
		child = item.data(0, CHILD_ROLE)
		if asset is None:
			return
		self.preview_gen += 1
		gen = self.preview_gen
		self.preview_title.setText(f"{asset.name}{' / ' + child if child else ''}  (rendering…)")
		project = self.project

		def work(_signals):
			try:
				return gen, asset, project.preview(asset, child), project.info(asset)
			except Exception as exc:	# noqa: BLE001 - keep the generation so stale errors are dropped
				return gen, asset, exc, None

		task = Task(work)
		task.signals.result.connect(self._preview_ready)
		task.signals.error.connect(self._preview_failed)
		self.pool.start(task)

	def _preview_ready(self, payload):
		gen, asset, pv, info = payload
		if gen != self.preview_gen:
			return	# a newer selection superseded this render
		if isinstance(pv, Exception):
			self._show_preview(None)
			self.preview_title.setText(f"Preview failed: {pv}")
			self.log(f"preview of {asset.name} failed: {pv}")
			return
		self._show_preview(pv, info, asset)

	def _preview_failed(self, tb):
		self.preview_title.setText("Preview failed")
		self.log(tb.strip().splitlines()[-1])

	def _show_preview(self, pv, info=None, asset=None):
		while self.inspector_form.rowCount():
			self.inspector_form.removeRow(0)
		if pv is None:
			self.preview_title.setText("Drop files or a gamedir here to begin.")
			self.src_pane.set_array(None, "Source")
			self.gx_pane.set_array(None, "GameCube (decoded)")
			return
		self.preview_title.setText(pv.title)
		def cap(label, arr):
			return f"{label}  {arr.shape[1]}×{arr.shape[0]}" if arr is not None else label
		self.src_pane.set_array(pv.source, cap("Source", pv.source))
		aspect = pv.source.shape[1] / pv.source.shape[0] if pv.source is not None else None
		gx_cap = cap("GameCube (decoded)", pv.result)
		if pv.result is not None and aspect and abs(aspect - pv.result.shape[1] / pv.result.shape[0]) > 1e-3:
			gx_cap += "  (power-of-two, shown at source aspect)"
		self.gx_pane.set_array(pv.result, gx_cap, aspect)
		self._apply_zoom(self.zoom.currentText())
		def section(title):
			lab = QLabel(title)
			lab.setStyleSheet("font-weight: bold; margin-top: 6px;")
			self.inspector_form.addRow(lab)
		section("Preview")
		for k, v in pv.info.items():
			self._info_row(k, v)
		if info:
			section("File")
			for k, v in info.items():
				self._info_row(k, v)
		if asset is not None and asset.warnings:
			section("Warnings")
			for w in asset.warnings:
				lab = QLabel(w)
				lab.setWordWrap(True)
				lab.setStyleSheet("color: #d9a63c;")
				self.inspector_form.addRow(lab)
		if asset is not None and asset.outputs:
			section("Outputs")
			for o in asset.outputs:
				self._info_row(o.rel, audit.human_bytes(o.size))

	def _info_row(self, k, v):
		val = QLabel(str(v))
		val.setWordWrap(True)
		val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		self.inspector_form.addRow(f"{k}:", val)

	def _apply_zoom(self, text):
		z = 0.0 if text == "Fit" else float(text.rstrip("×"))
		self.src_pane.canvas.set_zoom(z)
		self.gx_pane.canvas.set_zoom(z)

	# -- export ----------------------------------------------------------------

	def pick_output(self):
		d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
		if d:
			self.out_edit.setText(d)
			self.qsettings.setValue("output_dir", d)

	def open_output(self):
		path = Path(self.out_edit.text())
		path.mkdir(parents=True, exist_ok=True)
		QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

	def convert(self, selected_only: bool):
		if self.busy:
			return
		assets = self._selected_assets() if selected_only else list(self.project.assets)
		assets = [a for a in assets if a.kind in CONVERTIBLE]
		if not assets:
			QMessageBox.information(self, "GxBake", "Nothing convertible is selected.")
			return
		out = Path(self.out_edit.text()).expanduser()
		self.qsettings.setValue("output_dir", str(out))
		self.busy = True
		self._update_buttons()
		self.progress.setVisible(True)
		self.progress.setRange(0, len(assets))
		self.log(f"--- converting {len(assets)} asset(s) to {out}")
		project = self.project

		def work(signals):
			return project.convert(assets, out, log=signals.log.emit, progress=signals.progress.emit)

		task = Task(work)
		task.signals.log.connect(self.log)
		task.signals.progress.connect(self._on_progress)
		self._convert_out = out
		task.signals.result.connect(self._convert_done)
		task.signals.error.connect(self.log)
		task.signals.finished.connect(self._convert_finished)
		self.pool.start(task)

	def _on_progress(self, i, n, name):
		self.progress.setRange(0, max(n, 1))
		self.progress.setValue(i)
		self.statusBar().showMessage(f"Converting {name}…" if name else "")

	def _convert_done(self, written):
		out = self._convert_out
		for w in written:
			self.written[w.rel] = w.size
		total = sum(w.size for w in written)
		self.log(f"--- wrote {len(written)} file(s), {audit.human_bytes(total)} → {out}")
		self.statusBar().showMessage(f"Wrote {len(written)} files to {out}", 8000)

	def _convert_finished(self):
		self.busy = False
		self.progress.setVisible(False)
		for i in range(self.tree.topLevelItemCount()):
			item = self.tree.topLevelItem(i)
			self._fill_row(item, item.data(0, ASSET_ROLE))
		self._update_buttons()
		self._update_budget()
		self._preview_timer.start()

	def _update_budget(self):
		live = {o.rel for a in self.project.assets for o in a.outputs}
		outputs = {rel: size for rel, size in self.written.items() if rel in live}
		total = sum(outputs.values())
		if not outputs:
			self.budget.setText(f"{len(self.project.assets)} assets")
			return
		frac = total / audit.MAIN_RAM
		color = "#6c6" if frac < 0.25 else "#da3" if frac < 0.6 else "#e55"
		self.budget.setText(
			f"<span style='color:{color}'>GameCube output {audit.human_bytes(total)} "
			f"= {frac:.1%} of 24 MiB main RAM</span>")

	def _update_buttons(self):
		has = bool(self.project.assets)
		sel = bool(self.tree.selectedItems())
		for a in (self.a_add, self.a_add_dir):
			a.setEnabled(not self.busy)
		self.a_remove.setEnabled(sel and not self.busy)
		self.a_clear.setEnabled(has and not self.busy)
		self.a_convert_sel.setEnabled(sel and not self.busy)
		self.a_convert_all.setEnabled(has and not self.busy)
