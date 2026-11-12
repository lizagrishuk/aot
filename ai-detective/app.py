"""Streamlit entrypoint.

IMPORTANT: DO NOT replace this with ``from src.app import *``.
Streamlit reruns this entrypoint after every widget interaction.  A normal
Python import is cached, so that old wrapper executes the actual UI only once
and produces an empty page on all subsequent reruns.

runpy executes the real src/app.py on *every* Streamlit rerun, while keeping the
same Streamlit session and its widget state.
"""
from pathlib import Path
import runpy

UI = Path(__file__).resolve().parent / "src" / "app.py"
runpy.run_path(str(UI), run_name="__main__")
