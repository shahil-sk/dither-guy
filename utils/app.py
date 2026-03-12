# app.py
# Single entry point for Dither Guy.
#
# Run this file directly:
#   python app.py                          → launches the GUI
#   python app.py photo.jpg output.png     → headless CLI, single image
#   python app.py --batch in/ out/         → headless CLI, batch folder
#   python app.py --list-methods           → print available algorithms
#   python app.py --list-palettes          → print available palettes
#
# All heavy logic lives in utils/. This file only decides whether to
# start the GUI or the CLI based on sys.argv.

import sys

_CLI_FLAGS = {"--batch", "--list-methods", "--list-palettes"}

_has_cli = (
    len(sys.argv) > 1
    and (
        not sys.argv[1].startswith("-")
        or any(a in sys.argv for a in _CLI_FLAGS)
    )
)

if _has_cli:
    from utils.cli import cli
    cli()
else:
    from utils.app import main
    main()
