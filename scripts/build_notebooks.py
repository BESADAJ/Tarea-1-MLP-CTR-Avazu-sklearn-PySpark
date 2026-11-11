"""Convierte nb_src/*.py (celdas `# %%`) en book/*.ipynb y los ejecuta con el kernel de ml_venv."""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC, BOOK = ROOT / "nb_src", ROOT / "book"


def parse(text):
    cells = []
    for part in re.split(r"(?m)^# %%", text):
        if not part.strip():
            continue
        head, _, body = part.partition("\n")
        if "[markdown]" in head:
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.strip().splitlines())
            cells.append(new_markdown_cell(md))
        else:
            cells.append(new_code_cell(body.strip()))
    return cells


def build(path):
    nb = new_notebook(cells=parse(path.read_text(encoding="utf8")))
    nb.metadata["kernelspec"] = dict(name="python3", display_name="ml_venv", language="python")
    out = BOOK / (path.stem + ".ipynb")
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources=dict(metadata=dict(path=str(BOOK)))).execute()
    nbformat.write(nb, out)
    print("ok", out.name, flush=True)


targets = sys.argv[1:] or [p.stem for p in sorted(SRC.glob("*.py"))]
for t in targets:
    build(SRC / f"{t}.py")
