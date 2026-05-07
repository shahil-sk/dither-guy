from __future__ import annotations

# ── Palette (Adobe / After Effects inspired) ──────────────────────────────────
_P0 = "#141414"
_P1 = "#1a1a1a"
_P2 = "#222222"
_P3 = "#2a2a2a"
_P4 = "#333333"
_P5 = "#3d3d3d"
_P6 = "#555555"
_P7 = "#686868"

_G0 = "#4d9bff"
_G1 = "#2c7be5"
_G2 = "#1a5fbf"
_G3 = "#1a2e4a"

_AE  = "#ff8c00"
_AE2 = "#3d2200"

_AM = "#f0a500"
_RE = "#e84040"

_FG  = "#e8e8e8"
_FG2 = "#a0a0a0"
_FG3 = "#5a5a5a"
_FG4 = "#303030"

_MONO_FONT = "'Consolas', 'JetBrains Mono', 'Cascadia Code', 'Courier New', monospace"
_SANS_FONT = "'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif"

# ── Stylesheet ────────────────────────────────────────────────────────────────
_THEME = f"""
* {{ outline: none; }}

/* ── Global ──────────────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {_P1};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}

/* ── Toolbar ─────────────────────────────────────────────────────────────── */
QToolBar {{
    background: {_P0};
    border: none;
    border-bottom: 1px solid {_P5};
    padding: 3px 8px;
    spacing: 1px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 11px;
    color: {_FG2};
    font-family: {_SANS_FONT};
    font-size: 11px;
    letter-spacing: 0.2px;
    min-height: 24px;
}}
QToolBar QToolButton:hover {{
    color: {_FG};
    background: {_P4};
    border-color: {_P5};
}}
QToolBar QToolButton:pressed {{
    background: {_G3};
    color: {_G0};
    border-color: {_G2};
}}
QToolBar QToolButton:checked {{
    background: {_G3};
    border-color: {_G2};
    color: {_G0};
}}
QToolBar QLabel {{
    color: {_FG};
    font-family: {_SANS_FONT};
    font-weight: 600;
    font-size: 12px;
    padding: 0 14px 0 4px;
    letter-spacing: 0.5px;
}}
QToolBar::separator {{
    background: {_P5};
    width: 1px;
    margin: 5px 4px;
}}

/* ── Menu Bar ────────────────────────────────────────────────────────────── */
QMenuBar {{
    background: {_P0};
    color: {_FG2};
    font-family: {_SANS_FONT};
    font-size: 12px;
    border-bottom: 1px solid {_P5};
    padding: 1px 0;
}}
QMenuBar::item {{
    padding: 4px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {_P4};
    color: {_FG};
}}
QMenu {{
    background: {_P2};
    border: 1px solid {_P5};
    padding: 3px 0;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}
QMenu::item {{
    padding: 5px 28px 5px 18px;
}}
QMenu::item:selected {{
    background: {_G3};
    color: {_G0};
}}
QMenu::separator {{
    height: 1px;
    background: {_P5};
    margin: 3px 0;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {_P5};
    border-top: none;
    background: {_P1};
}}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: {_P0};
    border: 1px solid {_P5};
    border-bottom: none;
    padding: 7px 20px;
    font-family: {_SANS_FONT};
    font-size: 11px;
    font-weight: 400;
    color: {_FG3};
    margin-right: 1px;
    min-width: 80px;
}}
QTabBar::tab:selected {{
    background: {_P1};
    color: {_FG};
    border-bottom: 1px solid {_P1};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {_FG};
    background: {_P3};
}}

/* ── GroupBox ────────────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {_P5};
    border-radius: 2px;
    margin-top: 18px;
    padding-top: 18px;
    font-family: {_SANS_FONT};
    font-size: 10px;
    font-weight: 600;
    color: {_FG3};
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    font-size: 10px;
    left: 8px;
    top: -1px;
    padding: 0 4px;
    color: {_FG2};
    background: {_P2};
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ── Push Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 5px 14px;
    font-family: {_SANS_FONT};
    font-size: 12px;
    font-weight: 400;
    color: {_FG2};
    min-height: 24px;
}}
QPushButton:hover {{
    background: #3a3a3a;
    border-color: {_P6};
    color: {_FG};
}}
QPushButton:pressed {{
    background: {_P2};
    color: {_G0};
    border-color: {_G2};
}}
QPushButton:disabled {{
    background: {_P2};
    color: {_FG3};
    border-color: {_P3};
}}
QPushButton#accent {{
    background: {_G2};
    border: 1px solid {_G1};
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.2px;
    border-radius: 3px;
}}
QPushButton#accent:hover {{
    background: {_G1};
    border-color: {_G0};
    color: #ffffff;
}}
QPushButton#accent:pressed {{
    background: #164e99;
}}
QPushButton#danger {{
    background: #2b1111;
    border-color: #5a2020;
    color: {_RE};
    border-radius: 3px;
}}
QPushButton#danger:hover {{
    background: #3a1414;
    border-color: {_RE};
}}

/* ── ComboBox ────────────────────────────────────────────────────────────── */
QComboBox {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 4px 8px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    min-height: 24px;
    selection-background-color: {_G3};
}}
QComboBox:hover {{ border-color: {_P6}; background: #3a3a3a; }}
QComboBox:focus {{ border-color: {_G1}; }}
QComboBox::drop-down {{
    border: none;
    width: 20px;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {_FG2};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {_P3};
    border: 1px solid {_P5};
    selection-background-color: {_G3};
    selection-color: {_G0};
    padding: 2px 0;
    font-family: {_SANS_FONT};
    font-size: 12px;
    outline: 0;
}}

/* ── CheckBox ────────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 7px;
    font-size: 12px;
    color: {_FG2};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {_P6};
    border-radius: 2px;
    background: {_P4};
}}
QCheckBox::indicator:checked {{
    background: {_G1};
    border-color: {_G0};
}}
QCheckBox::indicator:hover {{ border-color: {_G1}; background: #3a3a3a; }}
QCheckBox::indicator:disabled {{
    background: {_P3};
    border-color: {_P5};
}}

/* ── Radio Button ────────────────────────────────────────────────────────── */
QRadioButton {{
    spacing: 7px;
    font-size: 12px;
    color: {_FG2};
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {_P6};
    border-radius: 7px;
    background: {_P4};
}}
QRadioButton::indicator:checked {{
    background: {_G1};
    border-color: {_G0};
}}

/* ── Slider ──────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {_P5};
    border-radius: 2px;
    margin: 0 2px;
}}
QSlider::sub-page:horizontal {{
    background: {_G0};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {_FG};
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
    border: 1px solid {_P6};
}}
QSlider::handle:horizontal:hover {{
    background: #ffffff;
    border-color: {_G0};
}}
QSlider::handle:horizontal:pressed {{
    background: {_G0};
    border-color: {_G0};
}}
QSlider:disabled::groove:horizontal {{
    background: {_P4};
}}
QSlider:disabled::handle:horizontal {{
    background: {_P6};
    border-color: {_P5};
}}

/* ── ScrollBar (wider, easier to grab) ───────────────────────────────────── */
QScrollArea {{ border: none; background: {_P0}; }}
QScrollBar:vertical {{
    background: {_P1};
    width: 10px;
    border-radius: 0px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: {_P1};
    height: 10px;
    border-radius: 0px;
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {_P6};
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{ background: #909090; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::corner {{ background: {_P0}; }}

/* ── ProgressBar ─────────────────────────────────────────────────────────── */
QProgressBar {{
    border: none;
    border-radius: 2px;
    background: {_P4};
    height: 5px;
    text-align: center;
    font-size: 10px;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {_G0};
    border-radius: 2px;
}}

/* ── StatusBar ───────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {_P0};
    color: {_FG3};
    font-family: {_MONO_FONT};
    font-size: 11px;
    border-top: 1px solid {_P5};
    padding: 2px 8px;
    min-height: 22px;
}}
QStatusBar QLabel {{ color: {_FG2}; font-size: 11px; }}

/* ── Splitter ────────────────────────────────────────────────────────────── */
QSplitter::handle:horizontal {{
    background: {_P5};
    width: 1px;
}}
QSplitter::handle:vertical {{
    background: {_P5};
    height: 1px;
}}
QSplitter::handle:hover {{
    background: {_G0};
}}

/* ── SpinBox ─────────────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 4px 6px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    min-height: 24px;
}}
QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {_P6}; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {_G1}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {_P3};
    width: 16px;
    border: none;
    border-left: 1px solid {_P5};
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {_P5};
}}

/* ── LineEdit ────────────────────────────────────────────────────────────── */
QLineEdit {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 4px 8px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    min-height: 24px;
    selection-background-color: {_G3};
    selection-color: {_G0};
}}
QLineEdit:focus {{
    border-color: {_G1};
    background: {_P3};
}}
QLineEdit:read-only {{
    color: {_FG3};
    background: {_P2};
}}

/* ── TextEdit / PlainTextEdit ────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background: {_P0};
    border: 1px solid {_P5};
    border-radius: 0px;
    color: {_FG};
    font-family: {_MONO_FONT};
    font-size: 12px;
    selection-background-color: {_G3};
    selection-color: {_G0};
    padding: 4px;
}}
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {_G1}; }}

/* ── List / Tree / Table ─────────────────────────────────────────────────── */
QListView, QTreeView, QTableView {{
    background: {_P2};
    border: 1px solid {_P5};
    border-radius: 0px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    alternate-background-color: {_P3};
    selection-background-color: {_G3};
    selection-color: {_G0};
    gridline-color: {_P5};
    outline: 0;
}}
QHeaderView::section {{
    background: {_P3};
    color: {_FG2};
    border: none;
    border-right: 1px solid {_P5};
    border-bottom: 1px solid {_P5};
    padding: 4px 8px;
    font-family: {_SANS_FONT};
    font-size: 11px;
    font-weight: 600;
}}
QHeaderView::section:hover {{ background: {_P4}; color: {_FG}; }}

/* ── Dialog ──────────────────────────────────────────────────────────────── */
QDialog {{
    background: {_P1};
    border: 1px solid {_P5};
}}

/* ── Frames ──────────────────────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {_P5};
}}

/* ── Tooltip ─────────────────────────────────────────────────────────────── */
QToolTip {{
    background: {_P3};
    border: 1px solid {_P6};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 11px;
    padding: 5px 9px;
    border-radius: 3px;
    opacity: 240;
}}

/* ── DockWidget ──────────────────────────────────────────────────────────── */
QDockWidget {{
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 11px;
    font-weight: 600;
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background: {_P3};
    border-bottom: 1px solid {_P5};
    padding: 4px 8px;
    text-align: left;
    letter-spacing: 0.4px;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 2px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {_P5};
}}

/* ── Label ───────────────────────────────────────────────────────────────── */
QLabel {{
    color: {_FG2};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}
QLabel[class="title"] {{
    color: {_FG};
    font-size: 13px;
    font-weight: 600;
}}

/* ── Accent panel strip ──────────────────────────────────────────────────── */
QGroupBox#accent_panel {{
    border-left: 2px solid {_G0};
    border-top: 1px solid {_P5};
    border-right: 1px solid {_P5};
    border-bottom: 1px solid {_P5};
}}
"""

# Legacy alias
THEME = _THEME
