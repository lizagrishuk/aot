import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import retrieval


def test_russian_stemmer_is_active():
    # без nltk top-15 отличаются от прогона в Colab и стресс-тесты теряют смысл
    assert retrieval._RU_STEMMER is not None, "установите nltk (pip install -r requirements.txt)"
    assert retrieval.tokenize("блокировка") == retrieval.tokenize("блокировки")


def test_missing_stemmer_fails_loudly(monkeypatch):
    import pytest
    monkeypatch.setattr(retrieval, "_RU_STEMMER", None)
    monkeypatch.delenv("ALLOW_NO_STEMMER", raising=False)
    with pytest.raises(RuntimeError, match="nltk"):
        retrieval.tokenize("блокировка")
