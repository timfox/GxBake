# PyInstaller spec: one-file GxBake executable (GUI by default, CLI with a subcommand).
#   pyinstaller packaging/gxbake.spec
# Drop an ffmpeg binary next to the executable (or set GXBAKE_FFMPEG) for video/audio.
import sys
from pathlib import Path

root = Path(SPECPATH).parent
excluded_qt = [
	"PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
	"PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
	"PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtSql", "PySide6.QtTest",
]

a = Analysis(
	[str(root / "gxbake" / "__main__.py")],
	pathex=[str(root)],
	# optional image decoders are imported on demand (texcodec.require), so name them; missing ones are skipped
	hiddenimports=["pillow_heif", "OpenEXR", "resvg_py", "psd_tools", "rawpy", "texture2ddecoder"],
	excludes=["tkinter", "matplotlib", "panda3d", "pillow_jxl", *excluded_qt],	# pillow_jxl is GPL-3.0	# .bam: bam2egg on PATH in frozen builds
	noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
	pyz,
	a.scripts,
	a.binaries,
	a.datas,
	name="GxBake",
	console=False if sys.platform == "win32" else True,
	upx=False,
)
if sys.platform == "darwin":
	app = BUNDLE(exe, name="GxBake.app", bundle_identifier="dev.gxbake.app")
