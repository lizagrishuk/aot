"""Create a clean archive for a new Colab run from the checked-out source."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lab6_project_clean.zip"
EXCLUDED_TOP = {".git", ".github", "results", ".pytest_cache", "__pycache__", ".venv", "venv"}
EXCLUDED_FILES = {"report.pdf", "report.md", OUT.name}

with ZipFile(OUT, "w", compression=ZIP_DEFLATED, compresslevel=7) as archive:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUT:
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_TOP for part in rel.parts):
            continue
        if path.name in EXCLUDED_FILES or path.name.startswith(".env") or path.suffix == ".pyc":
            continue
        archive.write(path, arcname=str(Path("ai-detective") / rel))
print(f"Created {OUT}")
