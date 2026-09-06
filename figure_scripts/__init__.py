import os
import sys
from pathlib import Path

_fig_dir: Path = Path(__file__).resolve().parent
_repo_root: Path = _fig_dir.parent

for _p in [str(_fig_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.exists("data") and (_repo_root / "data").exists():
    os.chdir(_repo_root)
