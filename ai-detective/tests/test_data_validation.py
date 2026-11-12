import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_data_validators_all_pass():
    for script in ("validate_documents.py","validate_claims.py","validate_questions.py"):
        proc=subprocess.run([sys.executable,str(ROOT/"tests"/script)],capture_output=True,text=True)
        assert proc.returncode==0,proc.stdout+proc.stderr
