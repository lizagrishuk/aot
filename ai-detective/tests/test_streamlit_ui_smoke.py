"""Headless widget-rerun smoke check, no browser or Ollama required.

A fake Streamlit surface drives 36 representative Q/mode/intervention UI reruns
in the actual src/app.py entrypoint. Browser networking is not simulated.
"""
from __future__ import annotations
import runpy
import sys
import types
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "app.py"


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.sidebar = self
        self.errors = []
        self.choice = {}
        self.resources = {}
        self.titles = 0

    def __enter__(self): return self
    def __exit__(self, *_): return False

    def cache_resource(self, **kwargs):
        def decorate(fn):
            def cached(*args, **kw):
                key = fn.__name__
                if key not in self.resources:
                    self.resources[key] = fn(*args, **kw)
                return self.resources[key]
            return cached
        return decorate

    def cache_data(self, **kwargs):
        return lambda fn: fn

    def form(self, *args, **kwargs): return self
    def expander(self, *args, **kwargs): return self
    def spinner(self, *args, **kwargs): return self
    def columns(self, amount, **kwargs): return [self for _ in range(amount)]

    def selectbox(self, label, options, **kwargs):
        key = kwargs.get('key')
        if key == 'question_select':
            return next(x for x in options if x.startswith(self.choice['qid'] + ' ·'))
        if key in self.choice:
            return self.choice[key]
        return options[0]

    def radio(self, label, options, **kwargs):
        return self.choice.get(kwargs.get('key'), options[0])

    def checkbox(self, *args, **kwargs): return False
    def file_uploader(self, *args, **kwargs): return None
    def form_submit_button(self, *args, **kwargs): return True
    def button(self, *args, **kwargs): return False
    def text_input(self, label, **kwargs): return ''
    def title(self, *args, **kwargs): self.titles += 1
    def error(self, *args, **kwargs): self.errors.append(' '.join(map(str,args)))
    def stop(self): raise AssertionError('The Streamlit script stopped unexpectedly')

    def __getattr__(self, name):
        # Non-interactive display primitives (text, dataframe, metric, caption, etc).
        return lambda *a, **kw: None


def test_real_streamlit_script_36_reactive_states(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    for ix in range(12):
        for mode in 'ABC':
            # Alternate between actual interventions and no-intervention.
            kind = ('none', 'permutation', 'add_contradiction')[ix % 3]
            fake.errors.clear()
            fake.choice = {
                'qid': f'Q{ix+1:02d}',
                'mode_select': next(v for v in (
                    'A — полный контекст', 'B — BM25 top-15', 'C — p-adic контекст') if v[0] == mode),
                'intervention_kind': kind,
                'section': 'Исследовательский стенд',
            }
            fake.session_state.clear()
            runpy.run_path(str(APP), run_name='__main__')
            assert not fake.errors, (fake.choice, fake.errors)
    # Exercise all official removal scenarios through the actual form.
    for number in range(3, 7):
        for mode in 'ABC':
            fake.errors.clear()
            fake.session_state.clear()
            fake.choice = {
                'qid': f'Q{number:02d}',
                'mode_select': next(v for v in (
                    'A — полный контекст', 'B — BM25 top-15', 'C — p-adic контекст') if v[0] == mode),
                'intervention_kind': 'remove_premise',
                'section': 'Исследовательский стенд',
            }
            runpy.run_path(str(APP), run_name='__main__')
            assert not fake.errors, (fake.choice, fake.errors)
    # Hand-selected fact removal must also render, and never present an old
    # archived response as the result of this newly modified prompt.
    fake.errors.clear()
    fake.session_state.clear()
    fake.choice = {
        'qid': 'Q01', 'mode_select': 'B — BM25 top-15',
        'intervention_kind': 'remove_premise', 'selected_removal_claim': 'C05',
        'section': 'Исследовательский стенд',
    }
    runpy.run_path(str(APP), run_name='__main__')
    assert not fake.errors
    # The separate comparison/archive screen must also render.
    fake.choice['section'] = 'Сравнение и архив'
    runpy.run_path(str(APP), run_name='__main__')
    assert not fake.errors


def test_root_entrypoint_reexecutes_ui_on_every_rerun(monkeypatch):
    """REGRESSION: old `from src.app import *` showed a blank second page."""
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    fake.choice = {
        'qid': 'Q01', 'mode_select': 'A — полный контекст',
        'intervention_kind': 'none', 'section': 'Исследовательский стенд',
    }
    root = APP.parent.parent / 'app.py'
    runpy.run_path(str(root), run_name='__main__')
    assert fake.titles == 1
    runpy.run_path(str(root), run_name='__main__')
    assert fake.titles == 2, 'The UI must be rebuilt after every Streamlit widget rerun'
    assert not fake.errors
