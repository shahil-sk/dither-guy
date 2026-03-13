from __future__ import annotations

# ── Palette ───────────────────────────────────────────────────────────────────
# _P0  = "#0f0f10"   # deepest background
# _P1  = "#161618"   # main background
# _P2  = "#1c1c1f"   # panel / sidebar background
# _P3  = "#222226"   # card / container
# _P4  = "#2a2a2e"   # toolbar / elevated surface
# _P5  = "#35353b"   # borders (normal)
# _P6  = "#48484f"   # borders (hover)

# # Accent — electric violet-blue
# _G0  = "#7c6dff"   # primary accent
# _G1  = "#6458e8"   # hover accent
# _G2  = "#4e44cc"   # active / pressed accent
# _G3  = "#312d80"   # muted accent / tint fills

# _AM  = "#f5a623"   # warning
# _RE  = "#ff4f5e"   # error

# _FG  = "#f2f2f5"   # primary text
# _FG2 = "#a0a0b0"   # secondary text
# _FG3 = "#5a5a6a"   # disabled / hint text

_P0  = "#1e1e1e"   # main background
_P1  = "#252526"   # panel background
_P2  = "#2d2d2d"   # secondary panel
_P3  = "#333333"   # card / container
_P4  = "#3c3c3c"   # toolbar
_P5  = "#464646"   # hover background
_P6  = "#555555"   # borders / separators

_G0  = "#3ea6ff"   # primary accent
_G1  = "#2b88d8"   # hover accent
_G2  = "#1f6fb2"   # active accent
_G3  = "#17558a"   # muted accent

_AM  = "#f5a623"   # warning
_RE  = "#e34850"   # error

_FG  = "#f0f0f0"   # primary text
_FG2 = "#b9b9b9"   # secondary text
_FG3 = "#7a7a7a"   # disabled / hint text

_MONO_FONT = "'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace"
_SANS_FONT = "'Inter', 'DM Sans', 'Outfit', 'Segoe UI', sans-serif"

# ── Stylesheet ────────────────────────────────────────────────────────────────
_THEME = f"""
* {{ box-sizing: border-box; outline: none; }}

QMainWindow, QWidget {{
    background-color: {_P1};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}

/* ── Toolbar ──────────────────────────────────────────────────────────────── */
QToolBar {{
    background: {_P0};
    border: none;
    border-bottom: 1px solid {_P5};
    padding: 4px 8px;
    spacing: 1px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 5px 11px;
    color: {_FG2};
    font-family: {_MONO_FONT};
    font-size: 11px;
    letter-spacing: 0.4px;
}}
QToolBar QToolButton:hover {{
    color: {_FG};
    background: {_P3};
    border-color: {_P5};
}}
QToolBar QToolButton:pressed {{
    background: {_P2};
    color: {_G0};
    border-color: {_G3};
}}
QToolBar QLabel {{
    color: {_G0};
    font-family: {_MONO_FONT};
    font-weight: bold;
    font-size: 13px;
    padding: 0 12px 0 4px;
    letter-spacing: 3px;
}}
QToolBar::separator {{
    background: {_P5};
    width: 1px;
    margin: 6px 4px;
}}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {_P5};
    border-top: none;
    background: {_P1};
    border-radius: 0 0 8px 8px;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: {_P2};
    border: 1px solid {_P5};
    border-bottom: none;
    padding: 8px 22px;
    font-family: {_SANS_FONT};
    font-size: 12px;
    font-weight: 500;
    color: {_FG3};
    # border-radius: 0px 0px 0 0;
    margin-right: 2px;
    min-width: 90px;
}}
QTabBar::tab:selected {{
    background: {_P1};
    color: {_G0};
    border-color: {_P5};
}}
QTabBar::tab:hover:!selected {{
    color: {_FG2};
    background: {_P3};
}}

/* ── GroupBox ─────────────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {_P5};
    border-radius: 2px;
    margin-top: 8px;
    padding-top: 14px;
    font-family: {_MONO_FONT};
    font-size: 9px;
    font-weight: bold;
    color: {_FG3};
    letter-spacing: 2px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    font-size: 12px;
    left: 10px;
    padding: 0 6px;
    color: {_G1};
    background: {_P1};
}}

/* ── Push Buttons ─────────────────────────────────────────────────────────── */
QPushButton {{
    background: {_P3};
    border: 1px solid {_P5};
    border-radius: 2px;
    padding: 6px 16px;
    font-family: {_SANS_FONT};
    font-size: 12px;
    font-weight: 500;
    color: {_FG2};
}}
QPushButton:hover {{
    background: {_P4};
    border-color: {_G3};
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
    background: {_G3};
    border-color: {_G2};
    color: {_G0};
    font-weight: 600;
}}
QPushButton#accent:hover {{
    background: {_G2};
    border-color: {_G0};
    color: #fff;
}}
QPushButton#danger {{
    background: #1c0a0d;
    border-color: #5a1a22;
    color: {_RE};
}}
QPushButton#danger:hover {{
    background: #2a0f14;
    border-color: {_RE};
}}

/* ── ComboBox ─────────────────────────────────────────────────────────────── */
QComboBox {{
    background: {_P3};
    border: 1px solid {_P5};
    border-radius: 2px;
    padding: 5px 10px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
    selection-background-color: {_G3};
}}
QComboBox:hover {{ border-color: {_G3}; background: {_P4}; }}
QComboBox:focus {{ border-color: {_G2}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {_FG2};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {_P2};
    border: 1px solid {_P5};
    selection-background-color: {_G3};
    selection-color: {_G0};
    padding: 4px 0;
    font-family: {_SANS_FONT};
    font-size: 12px;
    outline: 0;
}}

/* ── CheckBox ─────────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    font-size: 12px;
    color: {_FG2};
}}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {_P6};
    border-radius: 4px;
    background: {_P3};
}}
QCheckBox::indicator:checked {{
    background: {_G2};
    border-color: {_G0};
}}
QCheckBox::indicator:hover {{ border-color: {_G1}; background: {_P4}; }}

/* ── Slider ───────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {_P4};
    border-radius: 2px;
    margin: 0 3px;
}}
QSlider::sub-page:horizontal {{
    background: {_G1};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {_G0};
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid {_P1};
}}
QSlider::handle:horizontal:hover {{
    background: #9d91ff;
    border-color: {_G0};
}}

/* ── ScrollBar ────────────────────────────────────────────────────────────── */
QScrollArea {{ border: 1px solid {_P4}; background: {_P0}; }}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {_P1}; width: 6px; height: 6px; border-radius: 3px; margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {_P5}; border-radius: 3px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{ background: {_P6}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QScrollBar::corner {{ background: {_P1}; }}

/* ── ProgressBar ──────────────────────────────────────────────────────────── */
QProgressBar {{
    border: none;
    border-radius: 4px;
    background: {_P3};
    text-align: center;
    font-family: {_MONO_FONT};
    font-size: 9px;
    color: {_G0};
    height: 8px;
}}
QProgressBar::chunk {{
    background: {_G1};
    border-radius: 4px;
}}

/* ── StatusBar ────────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {_P0};
    color: {_FG3};
    font-family: {_MONO_FONT};
    font-size: 10px;
    border-top: 1px solid {_P5};
    padding: 3px 8px;
}}
QStatusBar QLabel {{ color: {_G1}; }}

/* ── Splitter ─────────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {_P5};
    width: 1px;
    height: 1px;
}}

/* ── SpinBox ──────────────────────────────────────────────────────────────── */
QSpinBox {{
    background: {_P3};
    border: 1px solid {_P5};
    border-radius: 6px;
    padding: 4px 8px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}
QSpinBox:hover {{ border-color: {_G3}; }}
QSpinBox:focus {{ border-color: {_G2}; }}
QSpinBox::up-button, QSpinBox::down-button {{
    background: {_P4};
    width: 18px;
    border-radius: 0;
    border: none;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {_P5}; }}

/* ── LineEdit ─────────────────────────────────────────────────────────────── */
QLineEdit {{
    background: {_P3};
    border: 1px solid {_P5};
    border-radius: 6px;
    padding: 5px 10px;
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}
QLineEdit:focus {{ border-color: {_G2}; background: {_P4}; }}

/* ── Dialog ───────────────────────────────────────────────────────────────── */
QDialog {{
    background: {_P1};
    border: 1px solid {_P5};
    border-radius: 10px;
}}

/* ── Frames ───────────────────────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {_P5}; }}

/* ── Tooltip ──────────────────────────────────────────────────────────────── */
QToolTip {{
    background: {_P0};
    border: 1px solid {_G3};
    color: {_FG2};
    font-family: {_SANS_FONT};
    font-size: 11px;
    padding: 5px 9px;
    border-radius: 6px;
    opacity: 230;
}}
"""

# Legacy alias kept for backwards compatibility
THEME = _THEME
