from __future__ import annotations

# ── Palette (Adobe / After Effects inspired) ──────────────────────────────────
# Adobe's professional dark UI uses a very specific gray family:
# near-black panels, precise neutral grays, and sharp blue/teal accents.
# No AMOLED pure black — Adobe uses #1a1a1a as the darkest surface.

# Surface layers — Adobe's characteristic warm-neutral dark grays
_P0 = "#141414"  # deepest background (like AE timeline bg)
_P1 = "#1a1a1a"  # main window base (Photoshop workspace gray)
_P2 = "#222222"  # panels / sidebars
_P3 = "#2a2a2a"  # cards / group boxes
_P4 = "#333333"  # elevated surfaces / inputs
_P5 = "#3d3d3d"  # borders (normal)
_P6 = "#555555"  # borders (hover)
_P7 = "#686868"  # disabled text bg hint

# Adobe Blue — PS/AE's signature cobalt accent
_G0 = "#4d9bff"  # primary accent (Adobe Blue, bright)
_G1 = "#2c7be5"  # hover state
_G2 = "#1a5fbf"  # active / pressed
_G3 = "#1a2e4a"  # muted fill / selection bg

# Secondary accent — After Effects orange-amber (used for timeline, warnings)
_AE = "#ff8c00"  # AE orange
_AE2 = "#3d2200" # AE orange muted bg

# Semantic
_AM = "#f0a500"  # warning — PS amber
_RE = "#e84040"  # error — AE red

# Text hierarchy — Adobe uses very controlled text weights
_FG  = "#e8e8e8"  # primary text (not pure white — softer)
_FG2 = "#a0a0a0"  # secondary / labels
_FG3 = "#5a5a5a"  # disabled / hints
_FG4 = "#303030"  # very subtle text on dark bg

# Fonts — Adobe uses clean, no-nonsense grotesques
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

/* ── Toolbar  (PS-style flat action bar) ─────────────────────────────────── */
QToolBar {{
    background: {_P0};
    border: none;
    border-bottom: 1px solid {_P5};
    padding: 2px 6px;
    spacing: 0px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    padding: 4px 10px;
    color: {_FG2};
    font-family: {_SANS_FONT};
    font-size: 11px;
    letter-spacing: 0.2px;
    min-height: 22px;
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
    margin: 5px 3px;
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

/* ── Tabs (PS panel tabs — compact, flat) ───────────────────────────────── */
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
    padding: 6px 18px;
    font-family: {_SANS_FONT};
    font-size: 11px;
    font-weight: 400;
    color: {_FG3};
    margin-right: 1px;
    min-width: 70px;
}}
QTabBar::tab:selected {{
    background: {_P1};
    color: {_FG};
    border-bottom: 1px solid {_P1};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {_FG2};
    background: {_P3};
}}

/* ── GroupBox (PS panel section headers) ─────────────────────────────────── */
QGroupBox {{
    border: 1px solid {_P5};
    border-radius: 0px;
    margin-top: 16px;
    padding-top: 18px;
    font-family: {_SANS_FONT};
    font-size: 11px;
    font-weight: 600;
    color: {_FG3};
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    font-size: 11px;
    left: 8px;
    top: -1px;
    padding: 0 4px;
    color: {_FG2};
    background: {_P1};
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

/* ── Push Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 2px;
    padding: 5px 14px;
    font-family: {_SANS_FONT};
    font-size: 12px;
    font-weight: 400;
    color: {_FG2};
    min-height: 22px;
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
/* Adobe-style primary CTA — solid blue */
QPushButton#accent {{
    background: {_G2};
    border: 1px solid {_G1};
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.2px;
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
}}
QPushButton#danger:hover {{
    background: #3a1414;
    border-color: {_RE};
}}

/* ── ComboBox (PS dropdown — flat, precise) ─────────────────────────────── */
QComboBox {{
    background: {_P4};
    border: 1px solid {_P5};
    border-radius: 2px;
    padding: 4px 8px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    min-height: 22px;
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
    width: 13px;
    height: 13px;
    border: 1px solid {_P6};
    border-radius: 2px;
    background: {_P4};
}}
QCheckBox::indicator:checked {{
    background: {_G1};
    border-color: {_G0};
    /* PS uses a filled square, not a checkmark glyph */
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
    width: 13px;
    height: 13px;
    border: 1px solid {_P6};
    border-radius: 7px;
    background: {_P4};
}}
QRadioButton::indicator:checked {{
    background: {_G1};
    border-color: {_G0};
}}

/* ── Slider (PS-style flat track) ───────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 3px;
    background: {_P5};
    border-radius: 1px;
    margin: 0 2px;
}}
QSlider::sub-page:horizontal {{
    background: {_G0};
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: {_FG};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
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

/* ── ScrollBar (ultra-slim, PS style) ───────────────────────────────────── */
QScrollArea {{ border: 1px solid {_P5}; background: {_P0}; }}
QScrollBar:vertical {{
    background: {_P1};
    width: 8px;
    border-radius: 0px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: {_P1};
    height: 8px;
    border-radius: 0px;
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {_P6};
    border-radius: 4px;
    min-height: 20px;
    min-width: 20px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{ background: #888888; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::corner {{ background: {_P0}; }}

/* ── ProgressBar ─────────────────────────────────────────────────────────── */
QProgressBar {{
    border: none;
    border-radius: 0px;
    background: {_P4};
    height: 4px;
    text-align: center;
    font-size: 10px;
    color: {_FG2};
}}
QProgressBar::chunk {{
    background: {_G0};
    border-radius: 0px;
}}

/* ── StatusBar (PS bottom bar) ───────────────────────────────────────────── */
QStatusBar {{
    background: {_P0};
    color: {_FG3};
    font-family: {_SANS_FONT};
    font-size: 11px;
    border-top: 1px solid {_P5};
    padding: 2px 8px;
    min-height: 20px;
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
    border-radius: 2px;
    padding: 4px 6px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    min-height: 22px;
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
    border-radius: 2px;
    padding: 4px 8px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    min-height: 22px;
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

/* ── List / Tree / Table views (PS layers / properties panels) ───────────── */
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
    background: {_P0};
    border: 1px solid {_P6};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 0px;
    opacity: 240;
}}

/* ── DockWidget (PS panel docks) ─────────────────────────────────────────── */
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

/* ── Accent panel strip (mimics AE blue left-border style) ──────────────── */
QGroupBox#accent_panel {{
    border-left: 2px solid {_G0};
    border-top: 1px solid {_P5};
    border-right: 1px solid {_P5};
    border-bottom: 1px solid {_P5};
}}
"""

# Legacy alias
THEME = _THEME
