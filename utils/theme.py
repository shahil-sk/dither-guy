from __future__ import annotations

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
_SANS_FONT = "'DM Sans', 'Outfit', 'Nunito', 'Segoe UI', sans-serif"

THEME = f"""
* {{ box-sizing: border-box; }}
QMainWindow, QWidget {{
    background-color: {_P1};
    color: {_FG};
    font-family: {_SANS_FONT};
    font-size: 12px;
}}
QToolBar {{
    background: {_P0};
    border-bottom: 1px solid {_G3};
    padding: 3px 6px;
    spacing: 2px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 5px 9px;
    color: {_FG2};
    font-family: {_MONO_FONT};
    font-size: 11px;
    letter-spacing: 0.5px;
}}
QToolBar QToolButton:hover {{
    color: {_G0};
    border-color: {_G3};
    background: {_P2};
}}
QToolBar QToolButton:pressed {{ background: {_P3}; color: {_G1}; }}
QToolBar QLabel {{
    color: {_G0};
    font-family: {_MONO_FONT};
    font-weight: bold;
    font-size: 13px;
    padding: 0 10px;
    letter-spacing: 1px;
}}
QTabWidget::pane {{
    border: 1px solid {_P5};
    border-top: 2px solid {_G3};
    background: {_P1};
}}
QTabBar::tab {{
    background: {_P2};
    border: 1px solid {_P4};
    border-bottom: none;
    padding: 7px 20px;
    font-family: {_MONO_FONT};
    font-size: 11px;
    color: {_FG3};
    letter-spacing: 0.5px;
    min-width: 80px;
}}
QTabBar::tab:selected {{
    background: {_P1};
    color: {_G0};
    border-top: 2px solid {_G0};
    border-bottom: none;
}}
QTabBar::tab:hover {{ color: {_FG2}; background: {_P3}; }}
QGroupBox {{
    border: 1px solid {_P5};
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 12px;
    font-family: {_MONO_FONT};
    font-size: 10px;
    font-weight: bold;
    color: {_FG3};
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 5px;
    color: {_G1};
    background: {_P1};
}}
QPushButton {{
    background: {_P3};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 6px 14px;
    font-family: {_MONO_FONT};
    font-size: 11px;
    color: {_FG2};
    letter-spacing: 0.3px;
}}
QPushButton:hover {{
    background: {_P4};
    border-color: {_G3};
    color: {_G0};
}}
QPushButton:pressed {{ background: {_P2}; color: {_G1}; }}
QPushButton:disabled {{
    background: {_P2};
    color: {_FG3};
    border-color: {_P3};
}}
QPushButton#accent {{
    background: {_G3};
    border-color: {_G1};
    color: {_G0};
}}
QPushButton#accent:hover {{
    background: {_G2};
    border-color: {_G0};
    color: #000;
}}
QPushButton#danger {{
    background: #1a0000;
    border-color: #660000;
    color: {_RE};
}}
QPushButton#danger:hover {{ background: #2a0000; border-color: {_RE}; }}
QComboBox {{
    background: {_P3};
    border: 1px solid {_P5};
    border-radius: 3px;
    padding: 5px 8px;
    color: {_FG};
    font-family: {_MONO_FONT};
    font-size: 11px;
}}
QComboBox:hover {{ border-color: {_G3}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {_FG2};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {_P2};
    border: 1px solid {_P5};
    selection-background-color: {_G3};
    selection-color: {_G0};
    padding: 2px;
    font-family: {_MONO_FONT};
    font-size: 11px;
}}
QCheckBox {{ spacing: 7px; font-size: 11px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {_P5};
    border-radius: 2px;
    background: {_P3};
}}
QCheckBox::indicator:checked {{
    background: {_G3};
    border-color: {_G1};
    image: none;
}}
QCheckBox::indicator:hover {{ border-color: {_G2}; }}
QSlider::groove:horizontal {{
    height: 3px;
    background: {_P4};
    border-radius: 1px;
    margin: 0 2px;
}}
QSlider::sub-page:horizontal {{
    background: {_G2};
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: {_G0};
    width: 12px; height: 12px;
    margin: -5px 0;
    border-radius: 2px;
    border: 1px solid {_G1};
}}
QSlider::handle:horizontal:hover {{ background: #33ff66; }}
QScrollArea {{ border: 1px solid {_P4}; background: {_P0}; }}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {_P1}; width: 6px; height: 6px; border-radius: 3px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {_P5}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{ background: {_P6}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QProgressBar {{
    border: none; border-radius: 2px;
    background: {_P3}; text-align: center;
    font-family: {_MONO_FONT}; font-size: 9px; color: {_G1};
}}
QProgressBar::chunk {{ background: {_G2}; border-radius: 2px; }}
QStatusBar {{
    background: {_P0};
    color: {_FG3};
    font-family: {_MONO_FONT};
    font-size: 10px;
    border-top: 1px solid {_G3};
    padding: 2px 6px;
}}
QStatusBar QLabel {{ color: {_G1}; }}
QSplitter::handle {{ background: {_P4}; width: 1px; height: 1px; }}
QSpinBox {{
    background: {_P3}; border: 1px solid {_P5};
    border-radius: 3px; padding: 4px 6px; color: {_FG};
    font-family: {_MONO_FONT}; font-size: 11px;
}}
QSpinBox:hover {{ border-color: {_G3}; }}
QSpinBox::up-button, QSpinBox::down-button {{
    background: {_P4}; width: 16px; border-radius: 2px; border: none;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {_P5}; }}
QLineEdit {{
    background: {_P3}; border: 1px solid {_P5};
    border-radius: 3px; padding: 5px 8px; color: {_FG};
    font-family: {_MONO_FONT}; font-size: 11px;
}}
QLineEdit:focus {{ border-color: {_G2}; }}
QDialog {{ background: {_P1}; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {_P5}; }}
QToolTip {{
    background: {_P0};
    border: 1px solid {_G3};
    color: {_G0};
    font-family: {_MONO_FONT};
    font-size: 10px;
    padding: 4px 7px;
    border-radius: 2px;
    opacity: 240;
}}
"""
