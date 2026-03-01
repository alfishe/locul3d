"""Reference point and coordinate display panel."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGridLayout, QFrame, QComboBox,
)
from PySide6.QtCore import Signal

from ...core.constants import AXIS_COLORS


def _axis_qcolor(axis):
    """Return CSS color string for axis index."""
    r, g, b = AXIS_COLORS[axis]
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


class ReferencePanel(QWidget):
    """Floating panel for reference point and coordinate mode."""

    set_ref_requested = Signal()
    clear_ref_requested = Signal()
    coord_mode_changed = Signal(int)  # 0=scene, 1=relative

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Reference point section
        lbl = QLabel("Reference Point")
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        self.btn_set_ref = QPushButton("Set (pick)")
        self.btn_set_ref.setToolTip("Click on point cloud to set reference origin")
        self.btn_set_ref.clicked.connect(lambda: self.set_ref_requested.emit())
        btn_row.addWidget(self.btn_set_ref)

        self.btn_clear_ref = QPushButton("Clear")
        self.btn_clear_ref.setToolTip("Reset to scene origin")
        self.btn_clear_ref.setEnabled(False)
        self.btn_clear_ref.clicked.connect(lambda: self.clear_ref_requested.emit())
        btn_row.addWidget(self.btn_clear_ref)
        layout.addLayout(btn_row)

        self.ref_label = QLabel("Origin: (scene)")
        self.ref_label.setStyleSheet("font-size: 11px; color: #aaa; padding: 2px 0;")
        layout.addWidget(self.ref_label)

        # Coordinate display section
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        lbl2 = QLabel("Coordinate Display")
        lbl2.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl2)

        self.coord_combo = QComboBox()
        self.coord_combo.addItems(["Scene (absolute)", "Relative (from ref)"])
        self.coord_combo.currentIndexChanged.connect(
            lambda idx: self.coord_mode_changed.emit(idx))
        layout.addWidget(self.coord_combo)

        # Reference point coordinates (read-only)
        grid = QGridLayout()
        grid.setSpacing(4)
        self.ref_spins = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            al = QLabel(f"{axis}:")
            al.setStyleSheet(f"color: {_axis_qcolor(i)};")
            grid.addWidget(al, i, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-10000, 10000)
            sp.setDecimals(3)
            sp.setReadOnly(True)
            sp.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            sp.setStyleSheet("background: transparent; border: 1px solid #444;")
            grid.addWidget(sp, i, 1)
            self.ref_spins[i] = sp
        layout.addLayout(grid)

        layout.addStretch()

    def set_ref_point(self, x, y, z):
        self.ref_label.setText(f"Origin: ({x:.3f}, {y:.3f}, {z:.3f})")
        self.btn_clear_ref.setEnabled(True)
        self.ref_spins[0].setValue(x)
        self.ref_spins[1].setValue(y)
        self.ref_spins[2].setValue(z)

    def clear_ref_point(self):
        self.ref_label.setText("Origin: (scene)")
        self.btn_clear_ref.setEnabled(False)
        for sp in self.ref_spins.values():
            sp.setValue(0.0)
