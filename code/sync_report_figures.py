"""Copy the figures used by the LaTeX reports into report/figures/.

Every \\includegraphics{figures/<name>} in report/*.tex is looked up in
results/*/figures/ and copied; figures in report/figures/ that no report uses
are removed, so the folder mirrors exactly what the reports need (and what the
Overleaf project links to).
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
DEST = REPORT / "figures"


def main():
    used = set()
    for tex in REPORT.glob("*.tex"):
        used |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}",
                               tex.read_text()))
    DEST.mkdir(exist_ok=True)
    for name in sorted(used):
        sources = list((ROOT / "results").glob(f"*/figures/{name}"))
        if not sources:
            raise FileNotFoundError(f"{name} is used in a report but not produced")
        shutil.copy(sources[0], DEST / name)
    removed = [f.name for f in DEST.iterdir() if f.name not in used]
    for name in removed:
        (DEST / name).unlink()
    print(f"report/figures: {len(used)} figures copied, {len(removed)} unused removed")


if __name__ == "__main__":
    main()
