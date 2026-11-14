"""Small Qt widgets: the texel viewer and a background-task runner."""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QObject, QRect, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


def to_qimage(arr: "np.ndarray") -> QImage:
	arr = np.ascontiguousarray(arr)
	if arr.ndim == 2:
		arr = np.dstack([arr, arr, arr, np.full(arr.shape, 255, np.uint8)])
	if arr.shape[2] == 3:
		arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
	h, w = arr.shape[:2]
	return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def _checker() -> QPixmap:
	pm = QPixmap(16, 16)
	pm.fill(QColor(70, 70, 70))
	p = QPainter(pm)
	p.fillRect(0, 0, 8, 8, QColor(100, 100, 100))
	p.fillRect(8, 8, 8, 8, QColor(100, 100, 100))
	p.end()
	return pm


class TexelCanvas(QWidget):
	"""Draws an image pixel-exact (nearest) over a checkerboard."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self._image: QImage | None = None
		self._zoom = 0.0	# 0 = fit
		self._aspect: float | None = None	# display aspect when texels are stretched
		self._checker = QBrush(_checker())
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

	def set_image(self, image: QImage | None, aspect: float | None = None) -> None:
		self._image = image
		self._aspect = aspect
		self._relayout()

	def _display_wh(self) -> tuple[float, float]:
		w, h = self._image.width(), self._image.height()
		if self._aspect and abs(self._aspect - w / h) > 1e-3:
			# Power-of-two textures are stretched; UVs are normalised, so show
			# them at the source's aspect, as the game will.
			return h * self._aspect, h
		return w, h

	def set_zoom(self, zoom: float) -> None:
		self._zoom = zoom
		self._relayout()

	def _relayout(self) -> None:
		if self._image is not None and self._zoom > 0:
			w, h = self._display_wh()
			self.setMinimumSize(QSize(int(w * self._zoom), int(h * self._zoom)))
		else:
			self.setMinimumSize(QSize(64, 64))
		self.update()

	def _target(self) -> QRect:
		dw, dh = self._display_wh()
		if self._zoom > 0:
			scale = self._zoom
		else:
			scale = min(self.width() / dw, self.height() / dh)
		w, h = int(dw * scale), int(dh * scale)
		return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

	def paintEvent(self, _event):	# noqa: N802 - Qt naming
		p = QPainter(self)
		p.fillRect(self.rect(), QColor(40, 40, 40))
		if self._image is None:
			p.setPen(QColor(150, 150, 150))
			p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
			return
		r = self._target()
		p.fillRect(r, self._checker)
		p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
		p.drawImage(r, self._image)


class ImagePane(QWidget):
	"""Caption + scrollable texel canvas."""

	def __init__(self, title: str, parent=None):
		super().__init__(parent)
		self.caption = QLabel(title)
		self.caption.setStyleSheet("font-weight: bold; padding: 2px;")
		self.canvas = TexelCanvas()
		self.scroll = QScrollArea()
		self.scroll.setWidget(self.canvas)
		self.scroll.setWidgetResizable(True)
		self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.addWidget(self.caption)
		lay.addWidget(self.scroll, 1)

	def set_array(self, arr, caption: str, aspect: float | None = None) -> None:
		self.caption.setText(caption)
		self.canvas.set_image(to_qimage(arr) if arr is not None else None, aspect)


class TaskSignals(QObject):
	result = Signal(object)
	error = Signal(str)
	log = Signal(str)
	progress = Signal(int, int, str)
	finished = Signal()


class Task(QRunnable):
	"""Runs fn(signals) on the thread pool; fn's return value is emitted as result."""

	def __init__(self, fn):
		super().__init__()
		self.fn = fn
		self.signals = TaskSignals()

	def run(self):
		try:
			self.signals.result.emit(self.fn(self.signals))
		except Exception:	# noqa: BLE001 - surface every failure in the UI
			self.signals.error.emit(traceback.format_exc())
		finally:
			self.signals.finished.emit()
